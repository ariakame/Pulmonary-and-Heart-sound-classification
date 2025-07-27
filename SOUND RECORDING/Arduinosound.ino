const int micPin = A1;
unsigned long lastMicros = 0;
const unsigned long sampleInterval = 62;  // ~16kHz (adjust if needed)

void setup() {
  Serial.begin(115200);
  analogRead(micPin);  // Dummy read to stabilize ADC
}

void loop() {
  if (micros() - lastMicros >= sampleInterval) {
    lastMicros += sampleInterval;

    int sample = analogRead(micPin);
    Serial.write(sample >> 2);  // Convert 10-bit to 8-bit
  }
}