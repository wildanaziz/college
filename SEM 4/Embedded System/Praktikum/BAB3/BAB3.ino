// // percobaan 1

// void setup() {
//   pinMode(LED_BUILTIN, OUTPUT);
// }

// void loop() {
//   digitalWrite(LED_BUILTIN, HIGH);
// }

// percobaan 2
// #include <LowPower.h>

// void setup() {
//   Serial.begin(9600);
//   pinMode(LED_BUILTIN, OUTPUT);
// }

// void loop() {
//   digitalWrite(LED_BUILTIN, HIGH);
//   delay(5000);
//   digitalWrite(LED_BUILTIN, LOW);
//   LowPower.powerExtStandby(SLEEP_8S, ADC_OFF, BOD_OFF, TIMER2_OFF);  
// }

// percobaan 3

// #include <LowPower.h>
// const int ledPin = 13;
// const int interruptPin = 2;

// void setup() {
//   pinMode(ledPin, OUTPUT);
//   pinMode(interruptPin, INPUT_PULLUP);
// }

// void flash(){
//   digitalWrite(ledPin, HIGH);
//   delay(500);
//   digitalWrite(ledPin, LOW);
//   delay(500);
//   digitalWrite(ledPin, HIGH);
//   delay(500);
//   digitalWrite(ledPin, LOW);
//   delay(500);
//   digitalWrite(ledPin, HIGH);
//   delay(500);
//   digitalWrite(ledPin, LOW);
//   delay(500);
// }


// void loop() {
// attachInterrupt(digitalPinToInterrupt(interruptPin), wakeUp, FALLING);
// LowPower.powerDown(SLEEP_FOREVER, ADC_OFF, BOD_OFF);
// detachInterrupt(0);
// flash();
// }

// void wakeUp() {

// }

// tugas no 1

// #include <LowPower.h>
// const int ledPin = 13;

// void setup() {
//   pinMode(ledPin, OUTPUT);
// }

// void flash(){
//   digitalWrite(ledPin, HIGH);
//   delay(500);
//   digitalWrite(ledPin, LOW);
//   delay(500);
//   digitalWrite(ledPin, HIGH);
//   delay(500);
//   digitalWrite(ledPin, LOW);
//   delay(500);
//   digitalWrite(ledPin, HIGH);
//   delay(500);
//   digitalWrite(ledPin, LOW);
//   delay(500);
// }


// void loop() {
//   LowPower.powerDown(SLEEP_8S, ADC_OFF, BOD_OFF);

//   flash();

// }

#include <LowPower.h>
#include <avr/wdt.h>

const int potPin = A0;
const int lampuKuning = 9;
const int lampuHijau = 8;
const int tombol = 2;

volatile bool triggerReset = false;

void setup() {
  pinMode(lampuKuning, OUTPUT);
  pinMode(lampuHijau, OUTPUT);
  pinMode(tombol, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(tombol), bangunkan, FALLING);

  Serial.begin(9600);
}

void loop() {
  if (triggerReset) {
    // Reset via watchdog
    Serial.println("Melakukan reset program...");
    delay(100); // beri waktu serial print
    wdt_enable(WDTO_15MS);
    while (1); // Tunggu reset
  }

  int nilaiAnalog = analogRead(potPin);
  int pwm = map(nilaiAnalog, 0, 1023, 0, 255);

  Serial.print("PWM: ");
  Serial.println(pwm);

  analogWrite(lampuKuning, pwm);
  delay(3000);
  analogWrite(lampuKuning, 0);

  if (tombol == HIGH) {
    Serial.print("aktif");
  }
  
  if (pwm > 240) {
    digitalWrite(lampuHijau, HIGH);
    Serial.println("Masuk Power Down");
    LowPower.powerDown(SLEEP_FOREVER, ADC_OFF, BOD_OFF);
    digitalWrite(lampuHijau, LOW);
  }

  
}

void bangunkan() {
  triggerReset = true;
}


