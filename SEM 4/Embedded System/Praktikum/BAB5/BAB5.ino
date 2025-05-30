#include <avr/wdt.h>

const int potPin = A0;      
const int ledPin = 13;      
int lastPotValue = -1;      
unsigned long lastChangeTime = 0; 
const unsigned long timeout = 1000; 

void setup() {
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, HIGH); 
  pinMode(potPin, INPUT);

  
  MCUSR &= ~(1 << WDRF);
  wdt_disable();

  Serial.begin(9600);
  Serial.println("= SYSTEM BOOTED");

  
  wdt_enable(WDTO_4S);
}

void loop() {
  int potValue = analogRead(potPin);
  int volume = map(potValue, 0, 1023, 1, 100);

  Serial.print("Volume: ");
  Serial.print(volume);
  Serial.println("%");

  
  if (potValue != lastPotValue) {
    lastPotValue = potValue;
    lastChangeTime = millis(); 
    digitalWrite(ledPin, HIGH); 
    wdt_reset(); 
  } else {
    if (millis() - lastChangeTime >= timeout) {
      digitalWrite(ledPin, LOW); 
      Serial.println("REBOOTED SYSTEM");
      Serial.flush(); 
      while (1); 
    }
  }

  // kalo volumenyya stabil maka nyalakan
  digitalWrite(ledPin, HIGH);

  
  wdt_reset();

  delay(100); 
}