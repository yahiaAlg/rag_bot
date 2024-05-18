import logging
from pprint import pprint
from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import *
from django.contrib.auth.decorators import login_required
import os
from .langchain_pipeline import get_api_keys, insert_or_create_index,searching_with_custom_prompt
# Create your views here.


from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import os
from .forms import UserChatForm
from .langchain_pipeline import searching_with_custom_prompt
from pprint import pprint

@login_required
def index(request):
    response = "no response"
    if request.method == "POST":
        form = UserChatForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data["chat_field"]
            print("query:\n", query)
            request.session["latest_question"] = query
            chunks = request.session.get("chunks", None)
            if chunks:
                try:
                    vector_store = insert_or_create_index("test-index", chunks)
                    pprint(vector_store)  # Debug: Print the vector store details

                    # Debug: Check if vector_store has the expected methods and attributes
                    logging.debug("Vector store methods: %s", dir(vector_store))

                    try:
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0) # type: ignore
                        response = llm.invoke(query)
                        del llm
                        pprint("response:\n", response) # type: ignore
                        request.session["latest_response"] = response
                    except Exception as e:
                        logging.error("Error invoking chain: %s", str(e))
                        messages.error(request, "There was an error processing your request.")
                except Exception as e:
                    logging.error("Error creating or inserting into index: %s", str(e))
                    messages.error(request, "Error preparing the vector store.")
            else:
                messages.error(request, "No document chunks available for search.")
    else:
        AVAILABLE_KEYS = [api_key for api_key in os.environ if "API_KEY" in api_key]
        print(AVAILABLE_KEYS)
        if not AVAILABLE_KEYS:
            api_key_msg = get_api_keys()
            bot_is_active = request.session.get("prepared", False)
            if api_key_msg and bot_is_active:
                messages.success(request, f'{api_key_msg}, you can ask your questions')
            elif not bot_is_active:
                messages.error(request, 'the bot is not active due to some internal error')
                return redirect("logout")
            else:
                messages.error(request, 'No API keys found, please add them to your environment variables')
                return redirect("logout")
        form = UserChatForm()
        query = request.session.get("latest_question", "No question asked yet")
        response = request.session.get("latest_response", "No response yet")

    context = {
        "user_chat_from": form,
        "response": response,
        "query": query,
    }
    return render(request, "pages/index.html", context)
