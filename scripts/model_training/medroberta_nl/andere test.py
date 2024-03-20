from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "CLTL/MedRoBERTa.nl"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {"question": "Ik heb een knelpunt in het midden van de borst en mijn hart lijkt te stoppen en dan te beginnen, help alstublieft. Ik vertelde dit aan mijn dokter en hij gaf me een röntgenfoto en de stresstest vond niets. Hij probeerde me depressiepillen te geven. Vertelde me dat ik angstaanvallen had. maar dit verbeeld ik mij niet", "context": "Graad begrijpt dat uw zorgen uw gegevens hebben doorgenomen. Angststoornis moet als basisstoornis worden beschouwd. Pijn op de borst kan te wijten zijn aan de zuurreflex die wordt veroorzaakt door de angststoornis, de daaruit voortvloeiende stress en uiteraard obsessie. Ik raad u aan eerst een arts te raadplegen en uw ECG te laten maken om te bevestigen dat alles perfect is met de gezondheid van uw hart. Dan kan een psycholoog uw angststoornis en daarmee uw pijn op de borst wegnemen. Als u op dit gebied meer van mijn hulp nodig heeft, kunt u deze URL gebruiken. http://goo.gl/aYW2pR. Zorg ervoor dat u elke minuut mogelijke details opneemt. Ik hoop dat dit uw vraag beantwoordt. Beschikbaar voor verdere toelichting. Succes."}

res = nlp(QA_input)

print(res)