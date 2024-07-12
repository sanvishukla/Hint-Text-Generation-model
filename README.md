# Hint-Text-Generation-model
This model is a tool designed to automate the generation and display of hint texts for EditText components on Android devices using a pre-trained language model. The model interacts with Android devices through ADB (Android Debug Bridge) to identify EditText components without hint texts. It leverages a GPT-2 language model to generate potential hints based on contextual information gathered from the device's UI hierarchy. 
## Setup Instructions
**Clone the Repository:**
   ```bash
   git clone https://github.com/sanvishukla/Hint-Text-Generation-model.git
   cd HintGenerator
```
**Install Dependencies:**
```bash
pip install -r requirements.txt
```
**Fine-tune the GPT-2 Model:**
<br> Run the trainmodel1.py to train the GPT-2 model on the given dataset, 80% of the dataset will be used to fine-tune and 20% will be used to calculate the metrics.<br><br>
**Connect Android Device:**
Connect your Android device to your computer via USB or using Wi-fi by typing the below command in terminal: <br>
Connect Android device using USB to your computer and open your IDE terminal.
   ```bash
adb tcpip 5555
adb connect [YOUR_IP_address]
   ```
You can access your IP Address by going to Settings> Additional Settings> Developer options> Wireless debugging.
<br><br>
**Run the Script:**
   ```bash
python hint_droid.py
   ```
