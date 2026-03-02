/*
  AGROPICK ACTUATOR FIRMWARE

  ESP32 responsibilities:
    - Rover motor actuation
    - Encoder readout
    - PCA9685 arm actuation
    - Deterministic USB serial protocol

  Serial protocol:
    READY|actuator-v1
    ACK|<id>
    DONE|<id>|ROVER
    DONE|<id>|POSE
    DONE|<id>|GRIPPER
    DONE|<id>|ROTGRIPPER
    DONE|<id>|HOME
    VAL|<id>|BUSY|0|1
    VAL|<id>|POS|base|shoulder|wrist|gripper|rotgripper
    VAL|<id>|ENC|left|right
    VAL|<id>|VERSION|actuator-v1
    ERR|<id>|<code>
*/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Rover pins
#define IN1 25
#define IN2 26
#define ENA 12
#define IN3 27
#define IN4 14
#define ENB 13

#define ENC1_A 34
#define ENC1_B 35
#define ENC2_A 32
#define ENC2_B 33

// PCA9685
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define BASE_CHANNEL 0
#define SHOULDER_CHANNEL 1
#define WRIST_CHANNEL 2
#define ROT_GRIPPER_CHANNEL 3
#define GRIPPER_CHANNEL 4
#define NUM_ARM_SERVOS 5

#define SERVOMIN 75
#define SERVOMAX 525

const char* FIRMWARE_VERSION = "actuator-v1";

const int BASE_MIN = 70;
const int BASE_MAX = 160;
const int SHOULDER_MIN = 120;
const int SHOULDER_MAX = 160;
const int WRIST_MIN = 150;
const int WRIST_MAX = 180;
const int GRIPPER_MIN = 30;
const int GRIPPER_MAX = 90;
const int ROT_GRIPPER_MIN = 130;
const int ROT_GRIPPER_MAX = 160;

const int HOME_BASE = 110;
const int HOME_SHOULDER = 150;
const int HOME_WRIST = 160;
const int HOME_ROT_GRIPPER = 130;
const int HOME_GRIPPER = 40;

const int SERVO_STEP = 2;
const int STEP_DELAY_MS = 12;

#define PWM_FREQ 1000
#define PWM_RESOLUTION 8

#define SERIAL_BUF_SIZE 96
#define ID_BUF_SIZE 20

struct ServoMove {
  bool active;
  uint8_t channel;
  int* currentPos;
  int targetAngle;
  unsigned long lastStepTime;
};

enum ArmActionKind {
  ARM_ACTION_NONE,
  ARM_ACTION_POSE,
  ARM_ACTION_GRIPPER,
  ARM_ACTION_ROTGRIPPER,
  ARM_ACTION_HOME
};

ServoMove servoMoves[NUM_ARM_SERVOS];

volatile long encoder1Count = 0;
volatile long encoder2Count = 0;

int basePos = HOME_BASE;
int shoulderPos = HOME_SHOULDER;
int wristPos = HOME_WRIST;
int gripperPos = HOME_GRIPPER;
int rotGripperPos = HOME_ROT_GRIPPER;

bool homeSequenceActive = false;
int homeStage = 0;

ArmActionKind activeArmAction = ARM_ACTION_NONE;
char activeArmActionId[ID_BUF_SIZE] = {0};

char serialBuf[SERIAL_BUF_SIZE];
int serialBufIdx = 0;

void IRAM_ATTR encoder1ISR() {
  encoder1Count += digitalRead(ENC1_B) ? 1 : -1;
}

void IRAM_ATTR encoder2ISR() {
  encoder2Count += digitalRead(ENC2_B) ? 1 : -1;
}

void copyId(char* dest, const char* source) {
  strncpy(dest, source, ID_BUF_SIZE - 1);
  dest[ID_BUF_SIZE - 1] = '\0';
}

int angleToPulse(int angle) {
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

void replyAck(const char* id) {
  Serial.print("ACK|");
  Serial.println(id);
}

void replyDone(const char* id, const char* kind) {
  Serial.print("DONE|");
  Serial.print(id);
  Serial.print("|");
  Serial.println(kind);
}

void replyValueBusy(const char* id) {
  Serial.print("VAL|");
  Serial.print(id);
  Serial.print("|BUSY|");
  Serial.println((homeSequenceActive || activeArmAction != ARM_ACTION_NONE) ? "1" : "0");
}

void replyValuePos(const char* id) {
  Serial.print("VAL|");
  Serial.print(id);
  Serial.print("|POS|");
  Serial.print(basePos);
  Serial.print("|");
  Serial.print(shoulderPos);
  Serial.print("|");
  Serial.print(wristPos);
  Serial.print("|");
  Serial.print(gripperPos);
  Serial.print("|");
  Serial.println(rotGripperPos);
}

void replyValueEnc(const char* id) {
  Serial.print("VAL|");
  Serial.print(id);
  Serial.print("|ENC|");
  Serial.print(encoder1Count);
  Serial.print("|");
  Serial.println(encoder2Count);
}

void replyValueVersion(const char* id) {
  Serial.print("VAL|");
  Serial.print(id);
  Serial.print("|VERSION|");
  Serial.println(FIRMWARE_VERSION);
}

void replyError(const char* id, const char* code) {
  Serial.print("ERR|");
  Serial.print(id);
  Serial.print("|");
  Serial.println(code);
}

bool isArmMotionActive() {
  for (int i = 0; i < NUM_ARM_SERVOS; i++) {
    if (servoMoves[i].active) {
      return true;
    }
  }
  return homeSequenceActive || activeArmAction != ARM_ACTION_NONE;
}

bool anyServoMoveActive() {
  for (int i = 0; i < NUM_ARM_SERVOS; i++) {
    if (servoMoves[i].active) {
      return true;
    }
  }
  return false;
}

void clearServoMoves() {
  for (int i = 0; i < NUM_ARM_SERVOS; i++) {
    servoMoves[i].active = false;
  }
}

void beginArmAction(ArmActionKind kind, const char* id) {
  activeArmAction = kind;
  copyId(activeArmActionId, id);
}

const char* armActionName(ArmActionKind kind) {
  switch (kind) {
    case ARM_ACTION_POSE:
      return "POSE";
    case ARM_ACTION_GRIPPER:
      return "GRIPPER";
    case ARM_ACTION_ROTGRIPPER:
      return "ROTGRIPPER";
    case ARM_ACTION_HOME:
      return "HOME";
    default:
      return "";
  }
}

void finishArmActionIfComplete() {
  if (activeArmAction == ARM_ACTION_NONE) {
    return;
  }
  if (homeSequenceActive || anyServoMoveActive()) {
    return;
  }
  replyDone(activeArmActionId, armActionName(activeArmAction));
  activeArmAction = ARM_ACTION_NONE;
  activeArmActionId[0] = '\0';
}

void cancelArmActions() {
  homeSequenceActive = false;
  homeStage = 0;
  clearServoMoves();
  activeArmAction = ARM_ACTION_NONE;
  activeArmActionId[0] = '\0';
}

void motor1Forward(int speed) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  ledcWrite(ENA, speed);
}

void motor1Backward(int speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  ledcWrite(ENA, speed);
}

void motor1Stop() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  ledcWrite(ENA, 0);
}

void motor2Forward(int speed) {
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  ledcWrite(ENB, speed);
}

void motor2Backward(int speed) {
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  ledcWrite(ENB, speed);
}

void motor2Stop() {
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  ledcWrite(ENB, 0);
}

void moveForward(int speed) {
  motor1Backward(speed);
  motor2Backward(speed);
}

void moveBackward(int speed) {
  motor1Forward(speed);
  motor2Forward(speed);
}

void turnLeft(int speed) {
  motor1Backward(speed);
  motor2Forward(speed);
}

void turnRight(int speed) {
  motor1Forward(speed);
  motor2Backward(speed);
}

void stopRover() {
  motor1Stop();
  motor2Stop();
}

bool validateBase(int angle) {
  return angle >= BASE_MIN && angle <= BASE_MAX;
}

bool validateShoulder(int angle) {
  return angle >= SHOULDER_MIN && angle <= SHOULDER_MAX;
}

bool validateWrist(int angle) {
  return angle >= WRIST_MIN && angle <= WRIST_MAX;
}

bool validateGripper(int angle) {
  return angle >= GRIPPER_MIN && angle <= GRIPPER_MAX;
}

bool validateRotGripper(int angle) {
  return angle >= ROT_GRIPPER_MIN && angle <= ROT_GRIPPER_MAX;
}

void startServoMove(uint8_t channel, int targetAngle, int &currentPos) {
  ServoMove &move = servoMoves[channel];
  move.active = true;
  move.channel = channel;
  move.currentPos = &currentPos;
  move.targetAngle = targetAngle;
  move.lastStepTime = millis();
}

void updateServoMoves() {
  unsigned long now = millis();

  for (int i = 0; i < NUM_ARM_SERVOS; i++) {
    ServoMove &move = servoMoves[i];
    if (!move.active) {
      continue;
    }
    if ((now - move.lastStepTime) < (unsigned long) STEP_DELAY_MS) {
      continue;
    }

    move.lastStepTime = now;
    int diff = move.targetAngle - *(move.currentPos);

    if (diff == 0) {
      move.active = false;
      continue;
    }

    int step = (diff > 0) ? SERVO_STEP : -SERVO_STEP;
    if (abs(diff) <= SERVO_STEP) {
      *(move.currentPos) = move.targetAngle;
    } else {
      *(move.currentPos) += step;
    }

    pwm.setPWM(move.channel, 0, angleToPulse(*(move.currentPos)));

    if (*(move.currentPos) == move.targetAngle) {
      move.active = false;
    }
  }
}

void moveServoBlocking(uint8_t channel, int targetAngle, int &currentPos) {
  if (currentPos == targetAngle) {
    pwm.setPWM(channel, 0, angleToPulse(currentPos));
    return;
  }
  int step = (currentPos < targetAngle) ? SERVO_STEP : -SERVO_STEP;
  while (currentPos != targetAngle) {
    if (abs(targetAngle - currentPos) <= SERVO_STEP) {
      currentPos = targetAngle;
    } else {
      currentPos += step;
    }
    pwm.setPWM(channel, 0, angleToPulse(currentPos));
    delay(STEP_DELAY_MS);
  }
}

void startHomeSequence() {
  homeSequenceActive = true;
  homeStage = 0;
  startServoMove(SHOULDER_CHANNEL, HOME_SHOULDER, shoulderPos);
}

void updateHomeSequence() {
  if (!homeSequenceActive) {
    return;
  }
  if (anyServoMoveActive()) {
    return;
  }

  homeStage++;
  switch (homeStage) {
    case 1:
      startServoMove(BASE_CHANNEL, HOME_BASE, basePos);
      break;
    case 2:
      startServoMove(WRIST_CHANNEL, HOME_WRIST, wristPos);
      break;
    case 3:
      startServoMove(ROT_GRIPPER_CHANNEL, HOME_ROT_GRIPPER, rotGripperPos);
      break;
    case 4:
      startServoMove(GRIPPER_CHANNEL, HOME_GRIPPER, gripperPos);
      break;
    default:
      homeSequenceActive = false;
      finishArmActionIfComplete();
      break;
  }
}

bool parseIntToken(const char* token, int* out) {
  if (token == nullptr || *token == '\0') {
    return false;
  }
  char* endPtr = nullptr;
  long value = strtol(token, &endPtr, 10);
  if (*endPtr != '\0') {
    return false;
  }
  *out = (int) value;
  return true;
}

void handlePing(const char* id) {
  replyAck(id);
}

void handleEstop(const char* id) {
  replyAck(id);
  stopRover();
  cancelArmActions();
}

void handleRover(const char* id, const char* direction, const char* speedToken) {
  int speed = 0;
  bool validDirection =
    strcmp(direction, "FWD") == 0 ||
    strcmp(direction, "REV") == 0 ||
    strcmp(direction, "LEFT") == 0 ||
    strcmp(direction, "RIGHT") == 0 ||
    strcmp(direction, "STOP") == 0;

  if (!validDirection) {
    replyError(id, "BAD_DIR");
    return;
  }

  if (strcmp(direction, "STOP") != 0) {
    if (!parseIntToken(speedToken, &speed)) {
      replyError(id, "BAD_SPEED");
      return;
    }
    speed = constrain(speed, 0, 255);
  }

  replyAck(id);

  if (strcmp(direction, "FWD") == 0) {
    moveForward(speed);
  } else if (strcmp(direction, "REV") == 0) {
    moveBackward(speed);
  } else if (strcmp(direction, "LEFT") == 0) {
    turnLeft(speed);
  } else if (strcmp(direction, "RIGHT") == 0) {
    turnRight(speed);
  } else {
    stopRover();
  }

  replyDone(id, "ROVER");
}

void handlePose(const char* id, const char* baseToken, const char* shoulderToken, const char* wristToken) {
  if (isArmMotionActive()) {
    replyError(id, "BUSY");
    return;
  }

  int base = 0;
  int shoulder = 0;
  int wrist = 0;
  if (!parseIntToken(baseToken, &base) || !parseIntToken(shoulderToken, &shoulder) || !parseIntToken(wristToken, &wrist)) {
    replyError(id, "BAD_POSE");
    return;
  }
  if (!validateBase(base) || !validateShoulder(shoulder) || !validateWrist(wrist)) {
    replyError(id, "POSE_RANGE");
    return;
  }

  replyAck(id);
  beginArmAction(ARM_ACTION_POSE, id);

  if (basePos != base) {
    startServoMove(BASE_CHANNEL, base, basePos);
  }
  if (shoulderPos != shoulder) {
    startServoMove(SHOULDER_CHANNEL, shoulder, shoulderPos);
  }
  if (wristPos != wrist) {
    startServoMove(WRIST_CHANNEL, wrist, wristPos);
  }

  finishArmActionIfComplete();
}

void handleGripper(const char* id, const char* angleToken) {
  if (isArmMotionActive()) {
    replyError(id, "BUSY");
    return;
  }

  int angle = 0;
  if (!parseIntToken(angleToken, &angle) || !validateGripper(angle)) {
    replyError(id, "GRIPPER_RANGE");
    return;
  }

  replyAck(id);
  beginArmAction(ARM_ACTION_GRIPPER, id);
  if (gripperPos != angle) {
    startServoMove(GRIPPER_CHANNEL, angle, gripperPos);
  }
  finishArmActionIfComplete();
}

void handleRotGripper(const char* id, const char* angleToken) {
  if (isArmMotionActive()) {
    replyError(id, "BUSY");
    return;
  }

  int angle = 0;
  if (!parseIntToken(angleToken, &angle) || !validateRotGripper(angle)) {
    replyError(id, "ROT_RANGE");
    return;
  }

  replyAck(id);
  beginArmAction(ARM_ACTION_ROTGRIPPER, id);
  if (rotGripperPos != angle) {
    startServoMove(ROT_GRIPPER_CHANNEL, angle, rotGripperPos);
  }
  finishArmActionIfComplete();
}

void handleHome(const char* id) {
  if (isArmMotionActive()) {
    replyError(id, "BUSY");
    return;
  }

  replyAck(id);
  beginArmAction(ARM_ACTION_HOME, id);
  startHomeSequence();
  finishArmActionIfComplete();
}

void handleGet(const char* id, const char* key) {
  bool validKey =
    strcmp(key, "BUSY") == 0 ||
    strcmp(key, "POS") == 0 ||
    strcmp(key, "ENC") == 0 ||
    strcmp(key, "VERSION") == 0;

  if (!validKey) {
    replyError(id, "BAD_GET");
    return;
  }

  replyAck(id);
  if (strcmp(key, "BUSY") == 0) {
    replyValueBusy(id);
  } else if (strcmp(key, "POS") == 0) {
    replyValuePos(id);
  } else if (strcmp(key, "ENC") == 0) {
    replyValueEnc(id);
  } else {
    replyValueVersion(id);
  }
}

void processLine(char* line) {
  char* tokens[6] = {nullptr};
  int tokenCount = 0;
  char* savePtr = nullptr;
  char* token = strtok_r(line, "|", &savePtr);

  while (token != nullptr && tokenCount < 6) {
    tokens[tokenCount++] = token;
    token = strtok_r(nullptr, "|", &savePtr);
  }

  if (tokenCount < 2) {
    return;
  }

  const char* command = tokens[0];
  const char* id = tokens[1];

  if (strcmp(command, "PING") == 0) {
    handlePing(id);
    return;
  }
  if (strcmp(command, "ESTOP") == 0) {
    handleEstop(id);
    return;
  }
  if (strcmp(command, "HOME") == 0) {
    handleHome(id);
    return;
  }

  if (strcmp(command, "ROVER") == 0) {
    if (tokenCount < 3) {
      replyError(id, "BAD_ROVER");
      return;
    }
    handleRover(id, tokens[2], (tokenCount >= 4) ? tokens[3] : nullptr);
    return;
  }

  if (strcmp(command, "POSE") == 0) {
    if (tokenCount < 5) {
      replyError(id, "BAD_POSE");
      return;
    }
    handlePose(id, tokens[2], tokens[3], tokens[4]);
    return;
  }

  if (strcmp(command, "GRIPPER") == 0) {
    if (tokenCount < 3) {
      replyError(id, "BAD_GRIPPER");
      return;
    }
    handleGripper(id, tokens[2]);
    return;
  }

  if (strcmp(command, "ROTGRIPPER") == 0) {
    if (tokenCount < 3) {
      replyError(id, "BAD_ROT");
      return;
    }
    handleRotGripper(id, tokens[2]);
    return;
  }

  if (strcmp(command, "GET") == 0) {
    if (tokenCount < 3) {
      replyError(id, "BAD_GET");
      return;
    }
    handleGet(id, tokens[2]);
    return;
  }

  replyError(id, "BAD_CMD");
}

void setup() {
  Serial.begin(115200);
  delay(250);

  for (int i = 0; i < NUM_ARM_SERVOS; i++) {
    servoMoves[i].active = false;
  }

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENC1_A, INPUT);
  pinMode(ENC1_B, INPUT);
  pinMode(ENC2_A, INPUT);
  pinMode(ENC2_B, INPUT);

  ledcAttach(ENA, PWM_FREQ, PWM_RESOLUTION);
  ledcAttach(ENB, PWM_FREQ, PWM_RESOLUTION);

  attachInterrupt(digitalPinToInterrupt(ENC1_A), encoder1ISR, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC2_A), encoder2ISR, RISING);

  stopRover();

  Wire.begin();
  Wire.setClock(400000);
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(50);

  moveServoBlocking(SHOULDER_CHANNEL, HOME_SHOULDER, shoulderPos);
  moveServoBlocking(BASE_CHANNEL, HOME_BASE, basePos);
  moveServoBlocking(WRIST_CHANNEL, HOME_WRIST, wristPos);
  moveServoBlocking(ROT_GRIPPER_CHANNEL, HOME_ROT_GRIPPER, rotGripperPos);
  moveServoBlocking(GRIPPER_CHANNEL, HOME_GRIPPER, gripperPos);

  Serial.print("READY|");
  Serial.println(FIRMWARE_VERSION);
}

void loop() {
  updateServoMoves();
  updateHomeSequence();
  finishArmActionIfComplete();

  while (Serial.available()) {
    char c = (char) Serial.read();

    if (c == '\n' || c == '\r') {
      if (serialBufIdx > 0) {
        serialBuf[serialBufIdx] = '\0';
        processLine(serialBuf);
        serialBufIdx = 0;
      }
    } else if (serialBufIdx < SERIAL_BUF_SIZE - 1) {
      serialBuf[serialBufIdx++] = c;
    } else {
      serialBufIdx = 0;
    }
  }
}
