def extract_relationships(concepts, text):
    """Extract more meaningful relationships between concepts"""
    relationships = []
    text_lower = text.lower()
    
    # Define relationship patterns
    built_with_patterns = [
        "built with", "made with", "developed with", "using", "based on", 
        "powered by", "implemented in", "written in"
    ]
    
    similar_to_patterns = [
        "similar to", "comparable to", "like", "resembles", "alternative to"
    ]
    
    depends_on_patterns = [
        "depends on", "requires", "needs", "dependency", "prerequisite"
    ]
    
    # Check for each concept pair
    concept_texts = [c["text"].lower() for c in concepts]
    
    for i, concept1 in enumerate(concepts):
        c1_text = concept1["text"].lower()
        
        for j, concept2 in enumerate(concepts):
            if i == j:
                continue
                
            c2_text = concept2["text"].lower()
            
            # Check for built_with relationship
            for pattern in built_with_patterns:
                search_text = f"{c1_text} {pattern} {c2_text}"
                if search_text in text_lower:
                    relationships.append({
                        "source": c1_text,
                        "target": c2_text,
                        "type": "BUILT_WITH"
                    })
            
            # Check for similar_to relationship
            for pattern in similar_to_patterns:
                search_text = f"{c1_text} {pattern} {c2_text}"
                if search_text in text_lower:
                    relationships.append({
                        "source": c1_text,
                        "target": c2_text,
                        "type": "SIMILAR_TO"
                    })
            
            # Check for depends_on relationship
            for pattern in depends_on_patterns:
                search_text = f"{c1_text} {pattern} {c2_text}"
                if search_text in text_lower:
                    relationships.append({
                        "source": c1_text,
                        "target": c2_text,
                        "type": "DEPENDS_ON"
                    })
    
    # Fall back to co-occurrence for relationships that weren't otherwise identified
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()
        paragraph_concepts = []
        
        for concept in concepts:
            if concept["text"].lower() in paragraph_lower:
                paragraph_concepts.append(concept)
        
        # Create relationships between all concepts in the same paragraph
        for i in range(len(paragraph_concepts)):
            for j in range(i+1, len(paragraph_concepts)):
                source = paragraph_concepts[i]["text"].lower()
                target = paragraph_concepts[j]["text"].lower()
                
                # Check if there's already a specific relationship
                if not any(r["source"] == source and r["target"] == target for r in relationships) and \
                   not any(r["source"] == target and r["target"] == source for r in relationships):
                    relationships.append({
                        "source": source,
                        "target": target,
                        "type": "CO_OCCURS_WITH"
                    })
    
    return relationships