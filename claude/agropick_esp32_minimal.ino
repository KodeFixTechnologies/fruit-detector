/*
  AGROPICK ESP32 - Minimal Actuator Firmware
  
  Pure serial-to-hardware bridge. No WiFi. No logging.
  Pi is the brain, this is the muscle.
  
  Protocol (115200 baud, newline-terminated):
  
  Arm:
    b:110       -> OK        set base
    s:130       -> OK        set shoulder
    w:160       -> OK        set wrist
    g:40        -> OK        set gripper
    r:130       -> OK        set rotgripper
    H           -> OK        home (OK when done)
    ?           -> B:110,S:130,W:160,G:40,R:130,M:0
  
  Rover:
    F120        -> OK        forward
    K120        -> OK        backward
    L100        -> OK        left
    T100        -> OK        turn right
    X           -> OK        stop
  
  Sensors:
    D           -> D:23.5    distance
    E           -> E:1234,-5678   encoders
    Z           -> OK        reset encoders

  Boot:
    RDY                      sent once on startup
*/

#include <ESP32Servo.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// PCA9685 channels
enum { CH_BASE, CH_SHLDR, CH_WRIST, CH_ROTG, CH_GRIP };

// Limits: {min, max, home}
static const int LIM[5][3] = {
  { 70, 160, 110},  // base
  {120, 160, 130},  // shoulder
  {150, 180, 160},  // wrist
  {130, 160, 130},  // rotgripper
  { 30,  90,  40},  // gripper
};

static int pos[5];
static int tgt[5];
static bool mov[5];
static unsigned long lstep[5];

#define STEP_MS 20
#define PWM_LO 75
#define PWM_HI 525

static bool homeRun = false;
static int homeStg = 0;

// Rover
#define IN1 25
#define IN2 26
#define ENA 12
#define IN3 27
#define IN4 14
#define ENB 13

#define ENC1A 34
#define ENC1B 35
#define ENC2A 32
#define ENC2B 33

volatile long enc1 = 0;
volatile long enc2 = 0;

#define TRIG 5
#define ECHO 18
#define SCAN_PIN 19

Servo scanServo;

#define BUF_SZ 32
static char buf[BUF_SZ];
static uint8_t bi = 0;

// ── Helpers ──

static inline int a2p(int a) { return map(a, 0, 180, PWM_LO, PWM_HI); }

static inline bool anyMov() {
  if (homeRun) return true;
  for (int i = 0; i < 5; i++) if (mov[i]) return true;
  return false;
}

static void startMov(int ch, int t) {
  if (t < LIM[ch][0] || t > LIM[ch][1]) return;
  if (pos[ch] == t) return;
  tgt[ch] = t;
  mov[ch] = true;
  lstep[ch] = millis();
}

static void tick() {
  unsigned long now = millis();
  for (int i = 0; i < 5; i++) {
    if (!mov[i]) continue;
    if (now - lstep[i] < STEP_MS) continue;
    lstep[i] = now;
    int d = tgt[i] - pos[i];
    if (d == 0) { mov[i] = false; continue; }
    pos[i] += (d > 0) ? 1 : -1;
    pwm.setPWM(i, 0, a2p(pos[i]));
  }
}

static void movBlk(int ch, int t) {
  int s = (pos[ch] < t) ? 1 : -1;
  while (pos[ch] != t) {
    if (abs(t - pos[ch]) <= 1) pos[ch] = t;
    else pos[ch] += s;
    pwm.setPWM(ch, 0, a2p(pos[ch]));
    delay(STEP_MS);
  }
}

static void tickHome() {
  if (!homeRun) return;
  for (int i = 0; i < 5; i++) if (mov[i]) return;
  static const int ord[5] = {CH_GRIP, CH_WRIST, CH_SHLDR, CH_BASE, CH_ROTG};
  if (homeStg < 5) {
    int c = ord[homeStg];
    startMov(c, LIM[c][2]);
    homeStg++;
  } else {
    homeRun = false;
    Serial.println("OK");
  }
}

static inline void m1(int a, int b, int s) {
  digitalWrite(IN1, a); digitalWrite(IN2, b); ledcWrite(ENA, s);
}
static inline void m2(int a, int b, int s) {
  digitalWrite(IN3, a); digitalWrite(IN4, b); ledcWrite(ENB, s);
}

void IRAM_ATTR isr1() { enc1 += digitalRead(ENC1B) ? 1 : -1; }
void IRAM_ATTR isr2() { enc2 += digitalRead(ENC2B) ? 1 : -1; }

// ── Command ──

static void proc() {
  if (bi == 0) return;
  buf[bi] = '\0';
  char c0 = buf[0];
  
  // Arm: b:110, s:130, w:160, g:40, r:130
  if (bi >= 3 && buf[1] == ':') {
    int v = atoi(buf + 2);
    int ch = -1;
    switch (c0) {
      case 'b': ch = CH_BASE;  break;
      case 's': ch = CH_SHLDR; break;
      case 'w': ch = CH_WRIST; break;
      case 'g': ch = CH_GRIP;  break;
      case 'r': ch = CH_ROTG;  break;
    }
    if (ch >= 0 && v >= LIM[ch][0] && v <= LIM[ch][1]) {
      startMov(ch, v);
      Serial.println("OK");
    } else {
      Serial.println("E");
    }
    return;
  }
  
  int spd = (bi > 1) ? atoi(buf + 1) : 150;
  if (spd < 0) spd = 0;
  if (spd > 255) spd = 255;
  
  switch (c0) {
    case 'H':
      homeRun = true; homeStg = 0;
      break;
    case '?':
      Serial.print("B:"); Serial.print(pos[0]);
      Serial.print(",S:"); Serial.print(pos[1]);
      Serial.print(",W:"); Serial.print(pos[2]);
      Serial.print(",G:"); Serial.print(pos[3]);
      Serial.print(",R:"); Serial.print(pos[4]);
      Serial.print(",M:"); Serial.println(anyMov() ? 1 : 0);
      break;
    
    // Rover
    case 'F': m1(0,1,spd); m2(0,1,spd); Serial.println("OK"); break;
    case 'K': m1(1,0,spd); m2(1,0,spd); Serial.println("OK"); break;
    case 'L': m1(0,1,spd); m2(1,0,spd); Serial.println("OK"); break;
    case 'T': m1(1,0,spd); m2(0,1,spd); Serial.println("OK"); break;
    case 'X': m1(0,0,0);   m2(0,0,0);   Serial.println("OK"); break;
    
    // Sensors
    case 'D': {
      digitalWrite(TRIG, LOW);  delayMicroseconds(2);
      digitalWrite(TRIG, HIGH); delayMicroseconds(10);
      digitalWrite(TRIG, LOW);
      long dur = pulseIn(ECHO, HIGH, 25000);
      Serial.print("D:");
      Serial.println((dur == 0) ? 999.0f : dur * 0.017f, 1);
      break;
    }
    case 'E':
      Serial.print("E:"); Serial.print(enc1);
      Serial.print(",");  Serial.println(enc2);
      break;
    case 'Z':
      enc1 = 0; enc2 = 0; Serial.println("OK"); break;
    
    // Scanner servo
    case 'V': {
      int a = (bi > 1) ? atoi(buf + 1) : 145;
      scanServo.write(constrain(a, 110, 180));
      Serial.println("OK");
      break;
    }
    
    default: Serial.println("E");
  }
}

// ── Setup ──

void setup() {
  Serial.begin(115200);
  
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(ENC1A, INPUT); pinMode(ENC1B, INPUT);
  pinMode(ENC2A, INPUT); pinMode(ENC2B, INPUT);
  pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT);
  
  ledcAttach(ENA, 1000, 8);
  ledcAttach(ENB, 1000, 8);
  
  attachInterrupt(digitalPinToInterrupt(ENC1A), isr1, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC2A), isr2, RISING);
  
  m1(0,0,0); m2(0,0,0);
  
  scanServo.attach(SCAN_PIN);
  scanServo.write(145);
  
  Wire.begin();
  Wire.setClock(400000);
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(50);
  
  for (int i = 0; i < 5; i++) {
    pos[i] = LIM[i][2];
    mov[i] = false;
  }
  movBlk(CH_GRIP, LIM[CH_GRIP][2]);
  movBlk(CH_WRIST, LIM[CH_WRIST][2]);
  movBlk(CH_SHLDR, LIM[CH_SHLDR][2]);
  movBlk(CH_BASE, LIM[CH_BASE][2]);
  movBlk(CH_ROTG, LIM[CH_ROTG][2]);
  
  Serial.println("RDY");
}

// ── Loop ──

void loop() {
  tick();
  tickHome();
  
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (bi > 0) { proc(); bi = 0; }
    } else if (bi < BUF_SZ - 1) {
      buf[bi++] = c;
    } else {
      bi = 0;
    }
  }
}
