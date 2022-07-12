

class Example(object):
    """A single training/test example for the roc dataset."""
    def __init__(self,
                 input_id,
                 context,
                 candidate,
                 ans = None,
                 adjacancy = None,
                 ask_for = None
                 ):
        self.input_id = input_id
        self.context = context
        self.candidate = candidate
        self.ans = ans - 1
        self.adjacancy = adjacancy
        self.ask_for = ask_for
       
        
class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 answer

    ):
        self.example_id = example_id
        try:
            self.choices_features = [
                {
                    'tokens': tokens,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'sentence_ind': sentence_ind,
                    'graph': graph,
                    'sentence_ids':graph_embedding
                }
                for tokens, input_ids, input_mask, segment_ids, sentence_ind, graph, graph_embedding in choices_features
            ]
        except: 
            self.choices_features = [
                {
                    'tokens': tokens,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'sentence_ind': sentence_ind,
                    'graph': graph
                }
                for tokens, input_ids, input_mask, segment_ids, sentence_ind, graph in choices_features
            ]   
        self.answer = answer
