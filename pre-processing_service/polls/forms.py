# polls/forms.py
from django import forms
from .models import Up_Fle

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = Up_Fle
        fields = ['file']

    def clean_file(self):
        up_file = self.cleaned_data.get("file")
        if not up_file.name.endswith('.csv'):
            raise forms.ValidationError("Разрешены только CSV-файлы.")
        return up_file
