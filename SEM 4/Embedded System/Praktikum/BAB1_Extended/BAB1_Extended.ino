#define tombol A0
char pinsCount = 6;
int pins[] = {3,5,6,9,10,11};
int val = 0;
volatile byte state = LOW;

void setup() {
  for (int i = 0; i<pinsCount; i++){
    pinMode(pins[i], OUTPUT);
  }
  pinMode(tombol, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(tombol), blink, CHANGE);
}

void loop() {
  val = digitalRead(tombol);
  blink();
  
}

void blink() {
  if (val == HIGH){
    for (int i = 0; i < pinsCount; i++){
      for (int j = 0; j <= 255; j++){
        analogWrite(pins[i], j); delay(3);
      }
      for (int j = 255; j >= 0; j--){
        analogWrite(pins[i], j); delay(3);
      }
    }
  }
  else if (val == LOW){
    for (int i = pinsCount - 1; i >= 0; i--){
      for (int j = 0; j <= 255; j++){
        analogWrite(pins[i], j); delay(3);
      }
      for (int j = 255; j >= 0; j--){
        analogWrite(pins[i], j); delay(1);
      }
    }
  }
}

// const byte ledPin = 13;

//     const byte interruptPin = 2;

//     volatile byte state = LOW;


//     void setup() {

//       pinMode(ledPin, OUTPUT);

//       pinMode(interruptPin, INPUT_PULLUP);

//       attachInterrupt(digitalPinToInterrupt(interruptPin), blink, CHANGE);

//     }


//     void loop() {

//       digitalWrite(ledPin, state);

//     }


    