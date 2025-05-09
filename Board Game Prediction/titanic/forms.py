# base/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User # If you need to customize fields or add email

class CustomUserCreationForm(UserCreationForm):
    # You can add email or other fields here if you want them on the registration form
    # email = forms.EmailField(required=True) # Example: making email required

    class Meta(UserCreationForm.Meta):
        model = User
        # Add 'email' to fields if you uncommented it above
        fields = UserCreationForm.Meta.fields # + ('email',) # Example
        # Or specify all fields explicitly:
        # fields = ("username", "email",) # etc.

# For a very basic registration, you can just use UserCreationForm directly in your view
# from django.contrib.auth.forms import UserCreationForm