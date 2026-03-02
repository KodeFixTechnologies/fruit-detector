/*
  AGROPICK UNIFIED CONTROLLER - Optimized
  
  Changes from original:
    - Non-blocking smooth servo movement (no delay() in moveServoSmooth)
    - Removed blocking getDistance() from status print loop
    - Reduced serial output to prevent buffer overflow
    - Added serial input buffer flush protection
    - Faster command processing loop
  
  WiFi: TomatoRover (192.168.4.1)
  
  Serial Commands:
    Rover: f120, b120, l100, r100, s
    Arm: base:110, shoulder:130, wrist:160, gripper:40, rotgripper:130, home
    Sensor: d (get distance on demand)
*/

#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ═══════════════════════════════════════════════════════════════════════════════
// WIFI CONFIGURATION (AP MODE)
// ═══════════════════════════════════════════════════════════════════════════════
const char* ap_ssid = "TomatoRover";
const char* ap_password = "tomato123";

IPAddress local_IP(192, 168, 4, 1);
IPAddress gateway(192, 168, 4, 1);
IPAddress subnet(255, 255, 255, 0);

WebServer server(80);

// ═══════════════════════════════════════════════════════════════════════════════
// PIN DEFINITIONS - ROVER
// ═══════════════════════════════════════════════════════════════════════════════
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

#define TRIG_PIN 5
#define ECHO_PIN 18

#define SERVO_PIN 19

// ═══════════════════════════════════════════════════════════════════════════════
// PCA9685 CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define BASE_CHANNEL 0
#define SHOULDER_CHANNEL 1
#define WRIST_CHANNEL 2
#define ROT_GRIPPER_CHANNEL 3
#define GRIPPER_CHANNEL 4

#define SERVOMIN 75
#define SERVOMAX 525

// Arm Limits
const int BASE_MIN = 70, BASE_MAX = 160;
const int SHOULDER_MIN = 120, SHOULDER_MAX = 160;
const int WRIST_MIN = 150, WRIST_MAX = 180;
const int GRIPPER_MIN = 30, GRIPPER_MAX = 90;
const int ROT_GRIPPER_MIN = 130, ROT_GRIPPER_MAX = 160;

// Arm Home Position
const int HOME_BASE = 110;
const int HOME_SHOULDER = 130;
const int HOME_WRIST = 160;
const int HOME_ROT_GRIPPER = 130;
const int HOME_GRIPPER = 40;

// Current arm positions
int basePos = HOME_BASE;
int shoulderPos = HOME_SHOULDER;
int wristPos = HOME_WRIST;
int gripperPos = HOME_GRIPPER;
int rotGripperPos = HOME_ROT_GRIPPER;

// ─── Non-blocking servo movement ───
// Each channel can have an active smooth move
struct ServoMove {
  bool active;
  int channel;
  int* currentPos;
  int targetAngle;
  unsigned long lastStepTime;
};

#define NUM_ARM_SERVOS 5
ServoMove servoMoves[NUM_ARM_SERVOS]; // one per channel

const int SERVO_STEP = 1;
const int STEP_DELAY_MS = 20;  // slightly faster than original 30ms

// Track if a "home" sequence is running
bool homeSequenceActive = false;
int homeStage = 0;

// ═══════════════════════════════════════════════════════════════════════════════
// ROVER OBJECTS & VARIABLES
// ═══════════════════════════════════════════════════════════════════════════════
Servo scanServo;

volatile long encoder1Count = 0;
volatile long encoder2Count = 0;

bool autonomousMode = false;
int roverSpeed = 150;

#define OBSTACLE_DISTANCE 30
#define SERVO_LEFT 180
#define SERVO_CENTER 145
#define SERVO_RIGHT 110

#define PWM_FREQ 1000
#define PWM_RESOLUTION 8

// Serial input buffer
#define SERIAL_BUF_SIZE 64
char serialBuf[SERIAL_BUF_SIZE];
int serialBufIdx = 0;

// ═══════════════════════════════════════════════════════════════════════════════
// ENCODER ISR
// ═══════════════════════════════════════════════════════════════════════════════
void IRAM_ATTR encoder1ISR() {
  encoder1Count += digitalRead(ENC1_B) ? 1 : -1;
}

void IRAM_ATTR encoder2ISR() {
  encoder2Count += digitalRead(ENC2_B) ? 1 : -1;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ULTRASONIC - Only called on demand now, not in status loop
// ═══════════════════════════════════════════════════════════════════════════════
float getDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  long duration = pulseIn(ECHO_PIN, HIGH, 25000); // reduced timeout
  return (duration == 0) ? 999.0f : duration * 0.034f / 2.0f;
}

void setServoAngle(int angle) {
  scanServo.write(constrain(angle, SERVO_RIGHT, SERVO_LEFT));
  delay(250);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROVER MOTOR CONTROL
// ═══════════════════════════════════════════════════════════════════════════════
void motor1Forward(int speed) {
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);  ledcWrite(ENA, speed);
}

void motor1Backward(int speed) {
  digitalWrite(IN1, HIGH);  digitalWrite(IN2, LOW);  ledcWrite(ENA, speed);
}

void motor1Stop() {
  digitalWrite(IN1, LOW);  digitalWrite(IN2, LOW);  ledcWrite(ENA, 0);
}

void motor2Forward(int speed) {
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);  ledcWrite(ENB, speed);
}

void motor2Backward(int speed) {
  digitalWrite(IN3, HIGH);  digitalWrite(IN4, LOW);  ledcWrite(ENB, speed);
}

void motor2Stop() {
  digitalWrite(IN3, LOW);  digitalWrite(IN4, LOW);  ledcWrite(ENB, 0);
}

void moveForward(int speed)  { motor1Backward(speed); motor2Backward(speed); }
void moveBackward(int speed) { motor1Forward(speed);  motor2Forward(speed);  }
void turnLeft(int speed)     { motor1Backward(speed); motor2Forward(speed);  }
void turnRight(int speed)    { motor1Forward(speed);  motor2Backward(speed); }
void stopRover()             { motor1Stop();          motor2Stop();          }

// ═══════════════════════════════════════════════════════════════════════════════
// ARM CONTROL - NON-BLOCKING
// ═══════════════════════════════════════════════════════════════════════════════
int angleToPulse(int angle) {
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

// Check if any servo is currently moving
bool isArmMoving() {
  for (int i = 0; i < NUM_ARM_SERVOS; i++) {
    if (servoMoves[i].active) return true;
  }
  return homeSequenceActive;
}

// Start a non-blocking smooth move on a channel
void startServoMove(int channel, int targetAngle, int &currentPos) {
  if (currentPos == targetAngle) return;
  
  // Clamp target immediately
  ServoMove &m = servoMoves[channel];
  m.active = true;
  m.channel = channel;
  m.currentPos = &currentPos;
  m.targetAngle = targetAngle;
  m.lastStepTime = millis();
}

// Tick all active servo moves - call this in loop()
void updateServoMoves() {
  unsigned long now = millis();
  
  for (int i = 0; i < NUM_ARM_SERVOS; i++) {
    ServoMove &m = servoMoves[i];
    if (!m.active) continue;
    
    if ((now - m.lastStepTime) >= (unsigned long)STEP_DELAY_MS) {
      m.lastStepTime = now;
      
      int diff = m.targetAngle - *(m.currentPos);
      if (diff == 0) {
        m.active = false;
        continue;
      }
      
      int step = (diff > 0) ? SERVO_STEP : -SERVO_STEP;
      if (abs(diff) <= SERVO_STEP) {
        *(m.currentPos) = m.targetAngle;
      } else {
        *(m.currentPos) += step;
      }
      
      pwm.setPWM(m.channel, 0, angleToPulse(*(m.currentPos)));
    }
  }
}

// Blocking version - only used during setup for initial home
void moveServoBlocking(int channel, int targetAngle, int &currentPos) {
  if (currentPos == targetAngle) return;
  int step = (currentPos < targetAngle) ? SERVO_STEP : -SERVO_STEP;
  while (currentPos != targetAngle) {
    if (abs(targetAngle - currentPos) < SERVO_STEP) {
      currentPos = targetAngle;
    } else {
      currentPos += step;
    }
    pwm.setPWM(channel, 0, angleToPulse(currentPos));
    delay(STEP_DELAY_MS);
  }
}

void setBase(int angle) {
  if (angle < BASE_MIN || angle > BASE_MAX) {
    Serial.println("ERR:base_range");
    return;
  }
  Serial.print("OK:base:");  Serial.println(angle);
  startServoMove(BASE_CHANNEL, angle, basePos);
}

void setShoulder(int angle) {
  if (angle < SHOULDER_MIN || angle > SHOULDER_MAX) {
    Serial.println("ERR:shoulder_range");
    return;
  }
  Serial.print("OK:shoulder:");  Serial.println(angle);
  startServoMove(SHOULDER_CHANNEL, angle, shoulderPos);
}

void setWrist(int angle) {
  if (angle < WRIST_MIN || angle > WRIST_MAX) {
    Serial.println("ERR:wrist_range");
    return;
  }
  Serial.print("OK:wrist:");  Serial.println(angle);
  startServoMove(WRIST_CHANNEL, angle, wristPos);
}

void setGripper(int angle) {
  if (angle < GRIPPER_MIN || angle > GRIPPER_MAX) {
    Serial.println("ERR:gripper_range");
    return;
  }
  Serial.print("OK:gripper:");  Serial.println(angle);
  startServoMove(GRIPPER_CHANNEL, angle, gripperPos);
}

void setRotGripper(int angle) {
  if (angle < ROT_GRIPPER_MIN || angle > ROT_GRIPPER_MAX) {
    Serial.println("ERR:rotgripper_range");
    return;
  }
  Serial.print("OK:rotgripper:");  Serial.println(angle);
  startServoMove(ROT_GRIPPER_CHANNEL, angle, rotGripperPos);
}

// Non-blocking home sequence managed in loop()
void startHomeSequence() {
  if (homeSequenceActive) return;
  Serial.println("OK:home_start");
  homeSequenceActive = true;
  homeStage = 0;
  // Stage 0: start gripper move
  startServoMove(GRIPPER_CHANNEL, HOME_GRIPPER, gripperPos);
}

void updateHomeSequence() {
  if (!homeSequenceActive) return;
  
  // Wait for current move to finish before starting next stage
  bool anyActive = false;
  for (int i = 0; i < NUM_ARM_SERVOS; i++) {
    if (servoMoves[i].active) { anyActive = true; break; }
  }
  if (anyActive) return; // still moving, wait
  
  homeStage++;
  switch (homeStage) {
    case 1:
      startServoMove(WRIST_CHANNEL, HOME_WRIST, wristPos);
      break;
    case 2:
      startServoMove(SHOULDER_CHANNEL, HOME_SHOULDER, shoulderPos);
      break;
    case 3:
      startServoMove(BASE_CHANNEL, HOME_BASE, basePos);
      break;
    case 4:
      startServoMove(ROT_GRIPPER_CHANNEL, HOME_ROT_GRIPPER, rotGripperPos);
      break;
    default:
      homeSequenceActive = false;
      Serial.println("OK:home_done");
      break;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMMAND PROCESSOR
// ═══════════════════════════════════════════════════════════════════════════════
void processCommand(const char* cmd, int len) {
  // Skip empty
  if (len == 0) return;
  
  // Trim whitespace
  while (len > 0 && (cmd[len-1] == ' ' || cmd[len-1] == '\t')) len--;
  while (len > 0 && (*cmd == ' ' || *cmd == '\t')) { cmd++; len--; }
  if (len == 0) return;
  
  // "home"
  if (len == 4 && (cmd[0] == 'h' || cmd[0] == 'H') && 
      (cmd[1] == 'o' || cmd[1] == 'O') && 
      (cmd[2] == 'm' || cmd[2] == 'M') && 
      (cmd[3] == 'e' || cmd[3] == 'E')) {
    startHomeSequence();
    return;
  }
  
  // "busy?" - let Pi query if arm is still moving
  if (len >= 4 && cmd[0] == 'b' && cmd[1] == 'u' && cmd[2] == 's' && cmd[3] == 'y') {
    Serial.println(isArmMoving() ? "BUSY" : "READY");
    return;
  }
  
  // "pos" - report current positions
  if (len >= 3 && cmd[0] == 'p' && cmd[1] == 'o' && cmd[2] == 's') {
    Serial.print("POS:");
    Serial.print(basePos); Serial.print(",");
    Serial.print(shoulderPos); Serial.print(",");
    Serial.print(wristPos); Serial.print(",");
    Serial.print(gripperPos); Serial.print(",");
    Serial.println(rotGripperPos);
    return;
  }
  
  // Arm commands (servo:angle)
  // Find colon
  int colonIdx = -1;
  for (int i = 0; i < len; i++) {
    if (cmd[i] == ':') { colonIdx = i; break; }
  }
  
  if (colonIdx > 0 && colonIdx < len - 1) {
    // Parse angle
    int angle = atoi(cmd + colonIdx + 1);
    
    // Match servo name (just check first 2-3 chars for speed)
    if (colonIdx >= 4 && cmd[0] == 'b' && cmd[1] == 'a') {
      setBase(angle);
    } else if (colonIdx >= 5 && cmd[0] == 's' && cmd[1] == 'h') {
      setShoulder(angle);
    } else if (colonIdx >= 5 && cmd[0] == 'w' && cmd[1] == 'r') {
      setWrist(angle);
    } else if (colonIdx >= 7 && cmd[0] == 'g' && cmd[1] == 'r') {
      setGripper(angle);
    } else if (colonIdx >= 4 && cmd[0] == 'r' && cmd[1] == 'o') {
      setRotGripper(angle);
    } else {
      Serial.println("ERR:servo");
    }
    return;
  }
  
  // Rover commands
  char command = cmd[0];
  int speed = roverSpeed;
  
  if (len > 1) {
    speed = atoi(cmd + 1);
    if (speed < 0) speed = 0;
    if (speed > 255) speed = 255;
  }
  
  switch (command) {
    case 'f': case 'F': autonomousMode = false; moveForward(speed); break;
    case 'b': case 'B': autonomousMode = false; moveBackward(speed); break;
    case 'l': case 'L': autonomousMode = false; turnLeft(speed); break;
    case 'r': case 'R': autonomousMode = false; turnRight(speed); break;
    case 's': case 'S': autonomousMode = false; stopRover(); break;
    case 'a': case 'A':
      autonomousMode = !autonomousMode;
      Serial.println(autonomousMode ? "AUTO:ON" : "AUTO:OFF");
      if (!autonomousMode) stopRover();
      break;
    case 'd': case 'D':
      // On-demand distance reading
      Serial.print("DIST:"); Serial.println(getDistance());
      break;
    case 'i': case 'I':
      Serial.print("CLIENTS:"); Serial.println(WiFi.softAPgetStationNum());
      break;
    default:
      Serial.println("ERR:cmd");
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WEB API HANDLERS
// ═══════════════════════════════════════════════════════════════════════════════
void handleRoot() {
  server.send(200, "text/html", "<h1>AgroPick</h1>");
}

void handleStatus() {
  // Lightweight JSON status
  String json = "{\"arm\":";
  json += isArmMoving() ? "\"busy\"" : "\"ready\"";
  json += ",\"base\":"; json += basePos;
  json += ",\"shoulder\":"; json += shoulderPos;
  json += ",\"wrist\":"; json += wristPos;
  json += ",\"gripper\":"; json += gripperPos;
  json += ",\"rotgripper\":"; json += rotGripperPos;
  json += ",\"enc1\":"; json += encoder1Count;
  json += ",\"enc2\":"; json += encoder2Count;
  json += ",\"clients\":"; json += WiFi.softAPgetStationNum();
  json += "}";
  server.send(200, "application/json", json);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SETUP
// ═══════════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(500);
  
  Serial.println("\n[AGROPICK UNIFIED - OPTIMIZED]");
  
  // Init servo move structs
  for (int i = 0; i < NUM_ARM_SERVOS; i++) {
    servoMoves[i].active = false;
  }
  
  // Rover pins
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(ENC1_A, INPUT); pinMode(ENC1_B, INPUT);
  pinMode(ENC2_A, INPUT); pinMode(ENC2_B, INPUT);
  pinMode(TRIG_PIN, OUTPUT); pinMode(ECHO_PIN, INPUT);
  
  scanServo.attach(SERVO_PIN);
  setServoAngle(SERVO_CENTER);
  
  ledcAttach(ENA, PWM_FREQ, PWM_RESOLUTION);
  ledcAttach(ENB, PWM_FREQ, PWM_RESOLUTION);
  
  attachInterrupt(digitalPinToInterrupt(ENC1_A), encoder1ISR, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC2_A), encoder2ISR, RISING);
  
  stopRover();
  Serial.println("OK:rover");
  
  // PCA9685
  Wire.begin();
  Wire.setClock(400000);  // Fast I2C - 400kHz instead of default 100kHz
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(50);
  Serial.println("OK:pca9685");
  
  // Blocking home only at boot
  moveServoBlocking(GRIPPER_CHANNEL, HOME_GRIPPER, gripperPos);
  moveServoBlocking(WRIST_CHANNEL, HOME_WRIST, wristPos);
  moveServoBlocking(SHOULDER_CHANNEL, HOME_SHOULDER, shoulderPos);
  moveServoBlocking(BASE_CHANNEL, HOME_BASE, basePos);
  moveServoBlocking(ROT_GRIPPER_CHANNEL, HOME_ROT_GRIPPER, rotGripperPos);
  Serial.println("OK:arm_home");
  
  // WiFi AP
  WiFi.mode(WIFI_AP);
  WiFi.softAPConfig(local_IP, gateway, subnet);
  if (WiFi.softAP(ap_ssid, ap_password)) {
    Serial.print("OK:wifi:");
    Serial.println(WiFi.softAPIP());
  } else {
    Serial.println("ERR:wifi");
  }
  
  // Web server
  server.on("/", handleRoot);
  server.on("/status", handleStatus);
  server.begin();
  
  Serial.println("READY");
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN LOOP - No blocking calls!
// ═══════════════════════════════════════════════════════════════════════════════
void loop() {
  // 1. Handle web clients
  server.handleClient();
  
  // 2. Update non-blocking servo movements
  updateServoMoves();
  
  // 3. Update home sequence if active
  updateHomeSequence();
  
  // 4. Process serial commands - fast char-by-char with buffer
  while (Serial.available()) {
    char c = (char)Serial.read();
    
    if (c == '\n' || c == '\r') {
      if (serialBufIdx > 0) {
        serialBuf[serialBufIdx] = '\0';
        processCommand(serialBuf, serialBufIdx);
        serialBufIdx = 0;
      }
    } else if (serialBufIdx < SERIAL_BUF_SIZE - 1) {
      serialBuf[serialBufIdx++] = c;
    } else {
      // Buffer overflow - discard
      serialBufIdx = 0;
    }
  }
  
  // 5. Minimal status print - every 5s, no blocking calls
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 5000) {
    lastPrint = millis();
    Serial.print("S:");
    Serial.print(encoder1Count); Serial.print(",");
    Serial.print(encoder2Count); Serial.print(",");
    Serial.print(WiFi.softAPgetStationNum()); Serial.print(",");
    Serial.println(isArmMoving() ? "M" : "R");  // M=moving, R=ready
  }
}
