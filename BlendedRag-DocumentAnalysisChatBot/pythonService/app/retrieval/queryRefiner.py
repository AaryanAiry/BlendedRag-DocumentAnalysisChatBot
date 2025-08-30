import re
from typing import List

_stopwords = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "in", "for", "to", "of", "by", "with", "as", "that", "this" 
}

_synonymMap = {
    "price": ["cost", "pricing"],
    "error": ["issue", "problem", "fault"],
    "summary": ["overview", "abstract"]
}

def basicPreprocess(query: str) -> str:
    q = query.strip().lower()
    q = re.sub(r'\s+',' ', q)
    return q

def expandQuery(query: str) -> List[str]:
    
    # Return a small set of query variants: 
    # cleaned + token-filtered + synonym

    refined = basicPreprocess(query)
    tokens = [t for t in re.split(r'\W+', refined) if t and t not in _stopwords]
    variants = [refined]

    #token synonyms
    for tok in tokens:
        if tok in _synonymMap:
            for syn in _synonymMap[tok]:
                variants.append(refined.replace(tok,syn))
    
    #add compact version
    if len(tokens) >1:
        variants.append(" ".join(tokens))
    
    #unqiue variants
    uniqVariants =[]
    for v in variants:
        if v not in uniqVariants:
            uniqVariants.append(v)
    return uniqVariants
    