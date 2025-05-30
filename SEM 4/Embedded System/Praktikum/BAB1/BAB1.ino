// percobaan 1

// char pinsCount = 6;
// int pins[] = {3, 5, 6, 9, 10, 11};
// void setup() {
//   for (int i = 0; i < pinsCount; i++){
//      pinMode(pins[i], OUTPUT);
//   }
// }

// void loop() {
//   for (int i = 0; i < pinsCount; i++){
//     digitalWrite(pins[i], HIGH);
//     delay(1000);
//     digitalWrite(pins[i], LOW);
//   }
// }

// percobaan 2

// #define tombol A0
// char pinsCount = 6;
// int pins[] = {3,5,6,9,10,11};
// int val = 0;

// void setup() {
//   for (int i = 0; i<pinsCount; i++){
//     pinMode(pins[i], OUTPUT);
//   }
//   pinMode(tombol, INPUT_PULLUP);
// }

// void loop() {
//   val = digitalRead(tombol);
//   if (val == HIGH){
//     for (int i = 0; i < pinsCount; i++){
//       for (int j = 0; j <= 255; j++){
//         analogWrite(pins[i], j); delay(3);
//       }
//       for (int j = 255; j >= 0; j--){
//         analogWrite(pins[i], j); delay(3);
//       }
//     }percobaan 2

…//         analogWrite(pins[i], j); delay(1);
//       }
//     }
//   }
// }

//   }
//   else if (val == LOW){
//     for (int i = pinsCount - 1; i >= 0; i--){
//       for (int j = 0; j <= 255; j++){
//         analogWrite(pins[i], j); delay(3);
//       }
//       for (int j = 255; j >= 0; j--){
//         analogWrite(pins[i], j); delay(1);
//       }
//     }
//   }
// }

// percobaan 3

// void setup() {
//   Serial.begin(9600);
// }

// void loop() {
//   int sensorValue = analogRead(A2);
//   Serial.println(sensorValue);
//   digitalWrite(8, HIGH);
//   delay(sensorValue);
//   digitalWrite(8, LOW);
//   delay(sensorValue);
// }