from django import forms


class VideoSearchForm(forms.Form):
    search = forms.CharField(max_length=100)
