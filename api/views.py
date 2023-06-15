from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd

classifier = TextClassifier.load('./api/ressources/final-model.pt')
df = pd.read_excel('./api/reponse.xlsx')
def label_type(label):
    value = label.split('_')
    return value[-1]
# Create your views here.


@api_view(['GET'])
def getData(request):
    question = request.GET.get('question')
    sentence = Sentence(question)
    classifier.predict(sentence)
    filtered_df = df[df['label'] == label_type(sentence.labels[0].value)]
    random_item = filtered_df.sample()
    column_value = random_item['text'].values[0]
    response = {"reponse": column_value}
    return Response(response)