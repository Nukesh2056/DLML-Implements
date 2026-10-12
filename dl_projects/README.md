

``` 
def create_data(text_data):
    toker.fit_on_texts([text_data])
    input_sequences = []
    
    # storing each sentence in 'sentences' variable
    for sentences in text_data.split(','):
        # converting the sentence into vector
        sentence_vector = toker.texts_to_sequences([sentences])[0]
        # converting the vectorised sentence into input format for model
        for i in range(1, len(sentence_vector)):
            input_sequences.append(sentence_vector[ : i+1])

```