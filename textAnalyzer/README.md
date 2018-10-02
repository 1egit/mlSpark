# **Text Analyzer**

The script provides four ways to analyze text data.

- The term frequency (TF):
  TF(w, F) = number of times the word w appears in document F.

-  The inverse document frequency (IDF):
    IDF(w, C) = log(Number of documents in corpus C / Number of documents in C containing word w)

- The TFIDF score:
  TFIDF(w, F, C) = TF(w, F) * IDF(w, C)

- The cosine similarity between two documents' TFIDF score

### **Usage**

Run the following from the command prompt:


- Look for help

 ```
  python TextAnalyzer.py -- help
 ```

- Compute TF
  ```
  python TextAnalyzer.py TF input output
  ```
  The program:
  - reads a text file F from argument `input`
  - compute the TF of each word in F
  - saves the result in `output`

- Find the words with highest frequency
  ```
  python TextAnalyzer.py TOP input output
  ```
  The program:
  - reads file `input`  of  the  (word, TF) pairs
  - find the pairs with 20 highest TF
  - saves the result in `output`

- Compute IDF
  ```
  python TextAnalyzer.py IDF input output
  ```
  The program:
  - reads a list of text file `input` representing corpus C
  - compute the IDF of each word in C
  - saves the result RDD in `output`

- Compute cosine similarity
  ```
  python TextAnalyzer.py SIM input output --other otherfile
  ```
  The program:
  - reads TFIDF scores of a text file from `input`
  - reads TFIDF scores of the other text file from `otherfile`
  - computes the cosine similarity of the two files
  - store the result in 'output'
