import spacy 

#Load a small English model for Spacy
nlp = spacy.load("en_core_web_sm")


def _read_data(filename):
    """	
    Read each row of a file
    """
    with open(filename, encoding="utf8") as f:
        matrix = []
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                yield matrix
                matrix = []
            else:
                if line[0] == '#':
                    continue
                matrix.append(line.split('\t'))
 
def read_data(filename):
    """	
    Read gold data readdata and return a list of sentences
    """    
    return list(_read_data(filename))


def pos_tagging(test_data):
    """
    Tag the text with pos tags
    """
    
    pred = []

    for sent in test_data:
        
        sent_str = ""
        for word in sent:
            sent_str += word[1] + " "
        
        sent_str = sent_str.rstrip()
        doc = nlp(sent_str)
        
        sent_pred = [(token.text, token.pos_) for token in doc]
        pred.append(sent_pred)

    print("Tagged sentences:", len(pred))
    return pred


def accuracy(gold_data, pred):
    """
    Evaluate how good the model is.
    """
    
    correct = total = 0
    
    for gold_sent, pred_sent in zip(gold_data, pred):
        for gold_word, pred_word in zip(gold_sent, pred_sent):
            total += 1
            if gold_word[3] == pred_word[1]:
                correct += 1

    if total == 0:
        return float('NaN')
    return (correct / total)


def dependency_tree(test_data):
    """
    Tag text with their dependency relation
    """
    
    pred = []
    sent_str = ""

    for sent in test_data:
        sent_str = ""
        for word in sent:
          sent_str += word[1] + " "
            
        sent_str = sent_str.rstrip()
        doc = nlp(sent_str)
        
        sent_pred = [(token.i, 
                      token.text, 
                      token.head.i, 
                      token.dep_.lower(), 
                      token.head.text) for token in doc]
        pred.append(sent_pred)

    print("Tagged sentences:", len(pred))
    return pred


def uas(gold_trees, pred_trees):
    """
    Evaluate result using Unlabled attachment score (UAS) 
    """
    
    n_total = n_correct = 0
    
    for gs, ps in zip(gold_trees, pred_trees):
        for g, p in zip(gs, ps):
            n_total += 1                
            if (g[7] == 'root' and p[3] == 'root'):
                 n_correct += 1
            else:
                n_correct += g[6] == str(p[2]+1)
    if n_total == 0:
        return float('NaN')
    else:
        return n_correct / n_total
    

def las(gold_trees, pred_trees):
    """
    Evaluate result using Labeled attachement score (LAS)
    """
    
    n_total = n_correct = 0
    
    for gs, ps in zip(gold_trees, pred_trees):
        for g, p in zip(gs, ps):
            n_total += 1
            if (g[7] == 'root' and p[3] == 'root'):
                n_correct += 1
            else:
                n_correct += g[6] == str(p[2]+1) and g[7] == p[3]
    if n_total == 0:
        return float('NaN')
    else:
        return n_correct / n_total