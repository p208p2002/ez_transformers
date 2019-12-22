class BertDataModel():
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer

    def toBertIds(self,input_a,input_b = None):
            tokenizer = self.tokenizer
            if(input_b is None):
                input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_a))
                return tokenizer.build_inputs_with_special_tokens(input_ids)
            else:
                input_a_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_a))
                input_b_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_b))
                return tokenizer.build_inputs_with_special_tokens(input_a_ids,input_b_ids)

            

    def add(self,input_a,input_b,label):
        '''
        input_a:str
        input_b
        label:str
        '''
        pass