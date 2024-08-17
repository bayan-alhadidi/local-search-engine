import os
import subprocess
import chardet
import langid
from flask import Flask, request, render_template, redirect, url_for
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Define the schema for the index
schema = Schema(title=ID(stored=True), path=ID(stored=True), content=TEXT(stored=True))

# Initialize or open the index
index_dir = os.path.join(os.path.dirname(__file__), 'indexdir')
if not os.path.exists(index_dir):
    os.mkdir(index_dir)
    ix = index.create_in(index_dir, schema)
else:
    if not index.exists_in(index_dir):
        ix = index.create_in(index_dir, schema)
    else:
        ix = index.open_dir(index_dir)

# Function to read .doc files using antiword
def read_doc_file(file_path):
    result = subprocess.run(['antiword', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        return result.stdout
    else:
        raise Exception(f"Error reading file: {result.stderr.decode()}")

# Function to get text from .doc files
def get_doc_text(filepath):
    
    if filepath.endswith('.doc'):
        content_bytes = read_doc_file(filepath)
        # Detect the encoding
        detection = chardet.detect(content_bytes)
        encoding = detection['encoding']
        try:
            content = content_bytes.decode(encoding)
        except (UnicodeDecodeError, TypeError):
            raise Exception(f"Failed to decode content with detected encoding: {encoding}")
        return content
    return ''

# Index the documents
def index_documents(directory, selected_language):
    def perform_indexing(writer, directory, selected_language):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.doc'):
                    filepath = os.path.join(root, file)
                    print(f"Indexing file: {filepath}")
                    content = get_doc_text(filepath)
                    try:
                        detected_language, _ = langid.classify(content)
                        if detected_language == selected_language:
                            writer.add_document(title=file, path=filepath, content=content)
                            print(f"Indexed file: {filepath}")
                        else:
                            print(f"Skipping file {filepath}, detected language: {detected_language}")
                    except Exception as e:
                        print(f"Language detection failed for file: {filepath} with error: {e}")
    
    try:
        with ix.writer() as writer:
            perform_indexing(writer, directory, selected_language)
    except index.LockError:
        print("Index is locked. Trying to clear the lock...")
        lockfile = os.path.join(index_dir, "WRITELOCK")
        if os.path.exists(lockfile):
            try:
                os.remove(lockfile)
                print("Lock file removed. Retrying...")
                with ix.writer() as writer:
                    perform_indexing(writer, directory, selected_language)
            except OSError as e:
                print(f"Error removing lock file: {e}")
        else:
            print("Lock file does not exist. Unable to clear lock.")


class BooleanModel:
    def __init__(self, documents):
        self.index = self.create_index(documents)

    def create_index(self, documents):
        index = {}
        for doc_id, doc in enumerate(documents):
            for term in doc.split():
                index.setdefault(term.lower(), set()).add(doc_id)
        return index

    def retrieve(self, query):
        query_terms = query.split()
        
        # Parsing the query to handle AND, OR, NOT
        current_set = set()
        operator = "AND"

        for term in query_terms:
            term = term.lower()
            if term == "AND" or term == "OR" or term == "NOT":
                operator = term
            else:
                term_set = self.index.get(term, set())
                if operator == "AND":
                    if current_set:
                        current_set &= term_set
                    else:
                        current_set = term_set
                elif operator == "OR":
                    current_set |= term_set
                elif operator == "NOT":
                    current_set -= term_set
        
        return current_set

class ExtendedBooleanModel:
    def __init__(self, documents):
        self.documents = documents
        self.index = self.create_index(documents)

    def create_index(self, documents):
        index = {}
        for doc_id, doc in enumerate(documents):
            for term in doc.lower().split():
                index.setdefault(term, set()).add(doc_id)
        return index

    def retrieve(self, query):
        terms = query.lower().split()
        results = {doc_id: 0 for doc_id in range(len(self.documents))}
        operators = {"AND", "OR", "NOT", "NEAR"}
        
        i = 0
        while i < len(terms):
            term = terms[i]

            if term not in operators:
                if i == 0 or terms[i-1] == "AND":
                    current_docs = self.index.get(term, set())
                elif terms[i-1] == "OR":
                    current_docs |= self.index.get(term, set())
                elif terms[i-1] == "NOT":
                    current_docs -= self.index.get(term, set())
                elif terms[i-1].startswith("NEAR/"):
                    proximity = int(terms[i-1].split('/')[1])
                    term_docs = self.index.get(term, set())
                    prev_term = terms[i-2]
                    prev_term_docs = self.index.get(prev_term, set())
                    current_docs = self.apply_proximity(prev_term, term, prev_term_docs, term_docs, proximity)
            else:
                i += 1
                continue
            
            for doc_id in current_docs:
                results[doc_id] += 1
            
            i += 1
        
        # Normalize scores
        max_score = max(results.values(), default=0)
        if max_score == 0:
            return set()
        
        threshold = 0.5  # Example threshold
        return {doc_id for doc_id, score in results.items() if score / max_score >= threshold}
    
    def apply_proximity(self, term1, term2, docs1, docs2, proximity):
        result_docs = set()
        for doc_id in docs1.intersection(docs2):
            positions1 = [i for i, word in enumerate(self.documents[doc_id].split()) if word.lower() == term1]
            positions2 = [i for i, word in enumerate(self.documents[doc_id].split()) if word.lower() == term2]
            for pos1 in positions1:
                for pos2 in positions2:
                    if abs(pos1 - pos2) <= proximity:
                        result_docs.add(doc_id)
                        break
        return result_docs
    
# Vector Space Model
class VectorSpaceModel:
    def __init__(self, documents):
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = self.vectorizer.fit_transform(documents)

    def retrieve(self, query, top_k=2):
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = np.dot(self.document_vectors, query_vector.T).toarray().flatten()
        top_docs = cosine_similarities.argsort()[-top_k:][::-1]
        return top_docs

def modify_path(path):
    return path.replace('\\', '/')

# Search the index
def search(query_str, model, documents, file_info):
    print(f"Searching with query: {query_str} using model: {model}")
    results = set()
    if model == "boolean":
        boolean_model = BooleanModel(documents)
        results = boolean_model.retrieve(query_str)
    elif model == "extended_boolean":
        extended_boolean_model = ExtendedBooleanModel(documents)
        results = extended_boolean_model.retrieve(query_str)
    elif model == "vector":
        vector_model = VectorSpaceModel(documents)
        top_docs = vector_model.retrieve(query_str)
        results = set(top_docs)
    
    result_files = [(file_info[doc_id]['title'], modify_path(file_info[doc_id]['path'])) for doc_id in results]
    return result_files

@app.route('/', methods=['GET', 'POST'])
def index_page():
    if request.method == 'POST':
        directory = request.form['directory']
        language = request.form['language']
        
        print(f"Indexing documents in directory: {directory} with language: {language}")
        index_documents(directory, language)
        
        # Save the settings to use later during search
        app.config['SEARCH_SETTINGS'] = {
            'directory': directory,
            'language': language
        }
        
        return redirect(url_for('search_page'))
    
    return render_template('index.html')


@app.route('/search', methods=['GET', 'POST'])
def search_page():
    results = []
    if request.method == 'POST':
        query = request.form['query']
        search_algorithm = request.form['search_algorithm']
        
        # Retrieve documents from the index
        documents = []
        file_info = []
        with ix.searcher() as searcher:
            for result in searcher.documents():
                documents.append(result['content'])
                file_info.append({'title': result['title'], 'path': result['path']})
        
        # Perform the search using the selected algorithm
        results = search(query, search_algorithm, documents, file_info)
        # Deduplicate results based on title (or any other unique identifier)
        unique_results = {item[0]: item for item in results}
        results = list(unique_results.values())
    else:
        query = ''
        search_algorithm = ''
    
    dir_path = modify_path(os.path.dirname(os.path.realpath(__file__)))
    return render_template('search.html', results=results, query=query, search_algorithm=search_algorithm, dir_path=dir_path)

@app.route('/preview')
def preview_file():
    path = request.args.get('path')
    if not path:
        return "No file path provided", 400
    try: 
        content = get_doc_text(path)
        return content
    except FileNotFoundError:
        return "File not found", 404
    except PermissionError:
        return "Permission denied", 403
    except Exception as e:
        return f"Error reading file: {e}", 500


if __name__ == '__main__':
    app.run(debug=True)
