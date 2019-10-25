import xml.etree.ElementTree as ET
import json

root = ET.parse("wsc.xml").getroot()

def strip_linebreaks(s):
    return s.replace('\r', '').replace('\n', '')

def strip_dots(s):
    return s.replace('.', '')

schemas = []

for schema in root:
    txt = schema.findall("text")
    assert len(txt) == 1
    txt = txt[0]

    infotext = strip_linebreaks(txt[0].text)
    infotext += "<key>"+strip_linebreaks(txt[1].text)+"</key>"
    infotext += strip_linebreaks(txt[2].text)

    quote = schema.findall("quote")
    assert len(quote) == 1
    quote = quote[0]

    question = ""
    offset = 0
    if len(quote) == 3:
        offset = 1
        if quote[0].text != None:
            t = quote[0].text.strip()
            if len(t) > 0:
                question += t
    question += "<key>"+strip_linebreaks(quote[offset].text).strip()+"</key>"
    question += strip_linebreaks(quote[offset+1].text)

    # create a single consistent whitespace around key span
    key_ind = question.find("<key>")
    if key_ind > 0:
        if question[key_ind-1] != " ":
            question = question[:key_ind]+ " "+question[key_ind:]
    key_ind = question.find("</key>")
    if key_ind + 6 < len(question):
        if question[key_ind+6] != " ":
            question = question[:key_ind+6]+ " "+question[key_ind+6:]

    answers = schema.findall("answers")
    assert len(answers) == 1
    answers = answers[0]
    answerA = answers[0].text.strip()
    answerB = answers[1].text.strip()

    correctAnswer = schema.findall("correctAnswer")
    assert len(correctAnswer) == 1
    correctAnswer = correctAnswer[0]
    correctAnswerTxt = strip_dots(correctAnswer.text.strip())

    schemas.append({
        "infotext": infotext.strip(),
        "question": question.strip(),
        "answerA": answerA.strip(),
        "answerB": answerB.strip(),
        "correctAnswer": correctAnswerTxt.strip()})

with open("wsc.json", "w") as f:
    json.dump(schemas, f)
