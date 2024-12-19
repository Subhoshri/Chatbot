# Bakerly, the Deep Learning Chatbot

## Introduction
This project is a contextual chatbot built using TensorFlow and inspired by the article [Contextual Chatbot With TensorFlow](https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077). The chatbot is designed to provide meaningful responses based on general input queries and also those related to Baking by leveraging natural language processing (NLP) techniques and machine learning.

## Features
- Tokenization and stemming of input text for preprocessing.
- Classification of user intents using bag-of-words.
- Training a neural network to classify intents.
- Generating appropriate responses based on user queries.
- Flexible and extensible design to add more intents and responses.

## Requirements
- Python 3.7 or higher
- Libraries:
  - TensorFlow
  - Numpy
  - NLTK
 
Refer `requirements.txt` for installing all the dependencies.


## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Subhoshri/Chatbot
   cd Chatbot
   ```

2. **Install required Libraries**:
   ```bash
   pip install Library-Name
   ```

3. **Prepare the Environment**:
   - Download NLTK data (if not already installed):
     ```python
     import nltk
     nltk.download('punkt')
     ```

4. **Update the intents file**:
   Modify the `intents.json` file to add or update chatbot intents and patterns.\

5. **Run the Application**:
   ```python
   python app.py
   ```

6. Enter the URL provided after running the previous commands into your web browser.

Bakerly is now ready to chat!


## How It Works
### 1. **Data Preprocessing**
   - The `intents.json` file contains training data in patterns, and their associated tags (and contexts).
   - Words in patterns are tokenized and stemmed using NLTK's Stemmer.
   - A bag-of-words representation is generated for each pattern.

### 2. **Model Training**
   - A feed-forward neural network (FFNN) is trained on the bag-of-words and output tags using TensorFlow.
   - The output is a one-hot-encoded vector representing the possible intent classes.

### 3. **Response Generation**
   - The trained model predicts the intent of user input.
   - The bot selects an appropriate response from the `responses` list associated with the predicted intent.

## Example Workflow
1. **User Input**: *"Hello!"*
2. **Preprocessing**:
   - Tokenize: `["hello"]`
   - Stem: `["hello"]`
   - Bag of Words: `[1, 0, 0, 0, ...]`
3. **Model Prediction**: Intent -> `greeting`
4. **Bot Response**: *"Hi there! How can I assist you today?"*

## Further Scope of Update
- **Adding New Intents**:
  - Update the `intents.json` file with new intents, patterns, and responses.
  - Retrain the model.

- **Improving Responses**:
  - Use additional NLP techniques, such as named entity recognition (NER) or sentiment analysis, to enhance contextual responses.


## Known Issues
- The chatbot may not handle out-of-scope queries effectively.
- Responses are limited to predefined data in the `intents.json` file.


## References
- [Chatbots Magazine Article](https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077)
- TensorFlow Documentation: [https://www.tensorflow.org](https://www.tensorflow.org)

---

## License
This project is licensed under the MIT License.
