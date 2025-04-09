from ninja_extra import NinjaExtraAPI
from main_ai import main_ai

api = NinjaExtraAPI()
sample_email_text = "michael pobega wrote i'm not sure if it's the mpl or mozilla that didn't allow the distribution of their images or the patching of programs without their knowledge but i think that is not dfsg free last time i looked the mozilla images were in an other licenses directory so not under the mpl and not licensed to others at all hope that helps mjr slef my opinion only see http people debian org mjr please follow"





@api.get("/email", tags=['Basic'])
def span_email_classifier(request, email_content: str =sample_email_text):

    data = {
        "email_content": email_content,
        "message": "Email content received",
        "AI_response": main_ai(request, email_content)
    }
    return {'status': 'success', "response":data}


