import os
import json
import math
import pickle
import random
import smtplib
import textwrap
from datetime import date, datetime, timedelta
from email.message import EmailMessage
from functools import reduce

import nltk
import numpy as np
import pywhatkit
import wikipedia
import pyttsx3
import speech_recognition as sr
from googlesearch import search
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MAIN CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("C:/Github_repos/Chatbot/intents.json").read())

words = pickle.load(open('C:/Github_repos/Chatbot/words.pkl', "rb"))
classes = pickle.load(open('C:/Github_repos/Chatbot/classes.pkl', "rb"))
model = load_model("C:/Github_repos/Chatbot/chatbot_model.model")


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for a, word in enumerate(words):
            if word == w:
                bag[a] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res_1 = model.predict(np.array([bow]), verbose=None)[0]
    error_threshold = 0.25
    results = [[b, r] for b, r in enumerate(res_1) if r > error_threshold]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# noinspection PyGlobalUndefined
def get_response(intents_list, intents_json):
    global bot_response
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for c in list_of_intents:
        if c['tag'] == tag:
            bot_response = random.choice(c['responses'])
            break
    return bot_response


def extract_from_text(n):
    my_list = []
    symbols = ["+", "-", "*", "**", "/", "^", "(", ")"]
    for t in n.split(' '):
        if t in symbols:
            my_list.append(t)
        else:
            try:
                my_list.append(float(t))
            except ValueError:
                pass

    return my_list


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OPERATIONS 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def youtube(n):
    yt_words = ["play", "on", "youtube", "yt"]
    yt_song = " ".join(d for d in n.split(" ") if d.lower() not in yt_words)
    yt_text = "Playing " + yt_song + " on YouTube...\n"
    print(yt_text)
    engine.say(yt_text)
    engine.runAndWait()
    pywhatkit.playonyt(yt_song)
    

def wiki(n):
    wiki_words = ["wikipedia", "wiki", "search", "for"]
    wiki_search = " ".join(e for e in n.split(" ") if e.lower() not in wiki_words)
    search_results = wikipedia.summary(wiki_search, 2)
    wiki_text = "Here's what I found on Wikipedia..."
    print(wiki_text)
    engine.say(wiki_text)
    engine.runAndWait()
    wrapper = textwrap.TextWrapper(width=140)
    word_list = wrapper.wrap(text=search_results)
    for element in word_list:
        print(element)


def google_search(n):
    google_words = ["google", "about", "search", "ask", "for", "to"]
    query = " ".join(f for f in n.split(" ") if f.lower() not in google_words)
    google_text = "Here are some links I found on google..."
    print(google_text)
    engine.say(google_text)
    engine.runAndWait()
    for j in search(query, num=5, stop=5, pause=2):
        print(j)
    print()


def website(_):
    url = input("Enter the website url: ")
    website_text = "Opening website..."
    print(website_text)
    engine.say(website_text)
    engine.runAndWait()
    os.system("start \"\" " + url)
    print()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OPERATIONS 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def fact(n):
    product = 1
    while n > 0:
        product = product * n
        n = n - 1
    return product


def evaluate(n):
    eval_exp = "".join(str(e) for e in n)
    answer = eval(eval_exp)
    eval_res = "The answer is: " + str(answer)
    return eval_res


def add(n):
    answer = str("The sum is : ") + str(reduce(lambda x, y: x + y, n))
    return answer


def sub1(n):
    answer = str(n[1]) + str(' - ') + str(n[0]) + str(' = ') + str(n[1] - n[0])
    return answer


def sub2(n):
    answer = str(n[0]) + str(' - ') + str(n[1]) + str(' = ') + str(n[0] - n[1])
    return answer


def multiply(n):
    answer = str("The product is : ") + str(reduce(lambda x, y: x * y, n))
    return answer


def divide(n):
    answer = str(n[0]) + str(' / ') + str(n[1]) + str(' = ') + str(n[0] / n[1])
    return answer


def sqrt(n):
    answer = str("Square root of ") + str(n[0]) + str(" is : ") + str(math.sqrt(n[0]))
    return answer


def exponent(n):
    answer = str(n[0]) + str(' ^ ') + str(n[1]) + str(' = ') + str(n[0] ** n[1])
    return answer


def log(n):
    if n[0] <= 0 or n[1] <= 0 or n[1] == 1:
        answer = str("Logarithm not defined")
    else:
        answer = str("Logarithm base ") + str(n[1]) + str(" of ") + str(n[0]) + str(" is : ") + str(math.log(n[0], n[1]))
    return answer


def ln(n):
    if n[0] <= 0:
        answer = str("Logarithm not defined")
    else:
        answer = str("Natural Log of ") + str(n[0]) + str(" is : ") + str(math.log(n[0], math.e))
    return answer


def factorial(n):
    if n[0] < 0:
        answer = str("Factorial not defined")
    else:
        answer = str(n[0]) + str('! = ') + str(fact(n[0]))
    return answer


def hcf(n):
    arr = np.array([int(x) for x in n])
    answer = str("The HCF is ") + str(np.gcd.reduce(arr))
    return answer


def lcm(n):
    arr = np.array([int(x) for x in n])
    answer = str("The LCM is ") + str(np.lcm.reduce(arr))
    return answer


def d_r(n):
    answer = str("The value of ") + str(n[0]) + str(" degrees in radians is : ") + str(math.radians(n[0]))
    return answer


def r_d(n):
    answer = str("The value of ") + str(n[0]) + str(" radians in degrees is : ") + str(math.degrees(n[0]))
    return answer


def sin(n):
    answer = str("The value of sin ") + str(n[0]) + str(" degrees is : ") + str(math.sin(math.radians(n[0])))
    return answer


def cos(n):
    answer = str("The value of cos ") + str(n[0]) + str(" degrees is : ") + str(math.cos(math.radians(n[0])))
    return answer


def tan(n):
    answer = str("The value of tan ") + str(n[0]) + str(" degrees is : ") + str(math.tan(math.radians(n[0])))
    return answer


def cosec(n):
    answer = str("The value of cosec ") + str(n[0]) + str(" degrees is : ") + str(1 / math.sin(math.radians(n[0])))
    return answer


def sec(n):
    answer = str("The value of sec ") + str(n[0]) + str(" degrees is : ") + str(1 / math.cos(math.radians(n[0])))
    return answer


def cot(n):
    answer = str("The value of cot ") + str(n[0]) + str(" degrees is : ") + str(1 / math.tan(math.radians(n[0])))
    return answer


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OPERATIONS 0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def end():
    input("Press enter key to exit")
    exit()


def date_today():
    today_date = date.today().strftime("%d/%m/%Y")
    date_res = "Today's date is: " + today_date
    return date_res


def day_today():
    today_day = datetime.now()
    day_res = "Today is " + today_day.strftime("%A")
    return day_res


def email(): 
    email_from = input("Enter your email address: ")
    email_password = input("Enter your password: ")
    email_to = input("Enter the receiver's email address: ")
    email_subject = input("Enter the subject of the email: ")
    email_body = input("Enter the contents of the email: ")
    my_email = EmailMessage()
    my_email["from"] = email_from
    my_email["to"] = email_to
    my_email["subject"] = email_subject
    my_email.set_content(email_body)
    with smtplib.SMTP(host='smtp.gmail.com', port=587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(email_from, email_password)
        smtp.send_message(my_email)
    email_res = "Email sent"
    return email_res


def whatsapp():
    whats_no = input("Enter the mobile number (with country code): ")
    whats_msg = input("Enter the message: ")
    updated = (datetime.now() + timedelta(seconds=20)).strftime('%H:%M:%S')
    time_list = [int(a) for a in updated.split(":")]
    hour_set = time_list[0]
    minute_set = time_list[1] + 1
    pywhatkit.sendwhatmsg(whats_no, whats_msg, hour_set, minute_set, wait_time=20)
    whatsapp_res = "Message sent"
    return whatsapp_res

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OPERATION DICTIONARIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
operations2 = {"youtube": youtube, "wiki": wiki, "google": google_search, "website": website}

# mathematical operations
operations1 = {"evaluate": evaluate, "add": add, "sub1": sub1, "sub2": sub2, "multiply": multiply, "divide": divide, "sqrt": sqrt, "exponent": exponent, "log": log, "ln": ln, "factorial": factorial, "hcf": hcf, "lcm": lcm, "d_r": d_r, "r_d": r_d, "sin": sin, "cos": cos, "tan": tan, "cosec": cosec, "sec": sec, "cot": cot}

operations0 = {"date": date_today, "day": day_today, "end": end, "email": email, "whatsapp": whatsapp}
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def main():
    start_text = "\nTARS is online\n"
    print(start_text)
    engine.say(start_text)
    engine.runAndWait()

    while True:
        message = input("")

        if message == "mic":
            listener = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    voice = listener.listen(source)
                    message = listener.recognize_google(voice)
                    print(message)
            except:
                pass

        ints = predict_class(message)
        res = get_response(ints, intents)

        try:
            if res in operations2:
                final_res = operations2[res](message)

            elif res in operations1:
                numbers = extract_from_text(message)
                final_res = operations1[res](numbers)

            elif res in operations0:
                final_res = operations0[res]()

            else:
                final_res = res

        except:
            final_res = "Task failed"
        
        if final_res != None:
                print(final_res, "\n")
                engine.say(final_res)
                engine.runAndWait()

engine = pyttsx3.init()
main()
