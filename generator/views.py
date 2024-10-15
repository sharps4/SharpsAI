from django.shortcuts import render
from django.views import View
# from ..manage import generate_text
from .utils import generate_text

class Index(View):
    def get(self, request):
        return render(request, "index.html")

    def post(self, request):
        text = request.POST.get('text')
        print(text)
        length = 20

        if not (text):
            return render(request, "index.html", {"error": "Merci d'écrire quelque chose"})

        result = generate_text(text, length)

        data = {
            "has_result":True,
            "result": result,
            "text": text,
        }
        return render(request, "index.html", data)

