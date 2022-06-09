from transformers import Wav2Vec2CTCTokenizer

class Wav2Vec2WordpieceTokenizer(Wav2Vec2CTCTokenizer):
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=False,
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            word_delimiter_token=word_delimiter_token,
            **kwargs,
        )

        self._create_trie(self.all_special_tokens_extended)
        
    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer.
        """
        special_cases = set(['gia', 'qui', 'quy', 'que', 'qua'])
        output_tokens = []
        for token_idx, token in enumerate(text.split()):
            if token in special_cases:
                sub_tokens = [token[:2], token[2:]]
            else:
                end = len(token)
                sub_tokens = []
                while end > 0:
                    start = 0
                    cur_substr = None
                    while start < end:
                        substr = token[start:end]
                        if substr in self.encoder:
                            cur_substr = substr
                            break
                        start += 1
                    if cur_substr is None:
                        sub_tokens.insert(0, self.unk_token)
                        end = start - 1
                    else:
                        sub_tokens.insert(0, cur_substr)
                        end = start
            
            if token_idx > 0:
                output_tokens.append(self.word_delimiter_token)
            output_tokens.extend(sub_tokens)
        return output_tokens
    
    def decode_ids(
        self, 
        token_ids, 
        skip_special_tokens = False, 
        clean_up_tokenization_spaces = True,
        group_tokens: bool = True,
        spaces_between_special_tokens: bool = False,
    ) -> str:
        # For compatible with speechbrain interfaces
        return self.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            group_tokens=group_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens
        )