from django import forms

class UserChatForm(forms.Form):
    chat_field = forms.CharField(
        label="",
        max_length=500,
        required=True,
        widget=forms.Textarea(attrs={"placeholder": "place your prompt here"}),
    )
