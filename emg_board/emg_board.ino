const int potPin = A0;  // Define the analog input pin for the potentiometer

void setup() {
  delay(10000);
  Serial.begin(9600);   // Initialize serial communication at 9600 bps
}

void loop() {
  
  int sensorValue = analogRead(potPin); // Read the value from the potentiometer
  //float voltage = (sensorValue / 1023.0) * voltageReference;

  // Send the sensor value to the Serial Monitor for reference
  // Send the sensor value to the Serial Plotter
  Serial.println(sensorValue);

  long sr = 1000/170;
  delay(10); // Optional delay for smoother readings, adjust as needed
}
