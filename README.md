# Chatbot
A simple chatbot made with tensorflow in python which can take natural language inputs to perform tasks such as:
1. Solve mathematical equations
2. Give information on day and date
3. Send whatsapp messages (using Pywhatkit library)
4. Send emails (using SMTP)
5. Perform google and wikipedia searches
6. Play songs on youtube

The model is trained on common phrases relevant to the above tasks, which are stored in the intents.json file. The training is done by running the training.py file.
The information on the phrases and their respective classes are dumped in a pkl file for the main program to read.

The model also accepts voice inputs using the pyttsx3 library.
