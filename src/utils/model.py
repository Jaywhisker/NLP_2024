# This is the python file containing the Roberta Class file
import torch
from transformers import RobertaModel

class RobertaClass(torch.nn.Module):
    def __init__(self,
                 dropout:float,
                 unfreeze_all:bool=False):
        """
        Custom RoBERTa model class that extends `torch.nn.Module`.

        Args:
            dropout (float): The dropout rate to apply during training to prevent overfitting.
            This will be applied after the RoBERTa model layers.
            unfreeze_all (bool, optional):
                If `True`, all layers of the RoBERTa model will be unfrozen (made trainable) for fine-tuning.
                If `False`, only the last layers (classifier layers) will be trainable, and the earlier layers will remain frozen. Default is `False`.
        """
        super(RobertaClass, self).__init__()
        #Roberta Base
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        # Classification layer for sentiment analysis
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(768, 2)

        if not unfreeze_all:
            # Freeze Roberta
            for param in self.l1.parameters():
                param.requires_grad = False
        else:
            # Unfreeze all
            for param in self.l1.parameters():
                param.requires_grad = True


    def forward(self, input_ids, attention_mask, token_type_ids):
        #Pass inputs through the pre-trained RoBERTa model
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Extract the [CLS] token's hidden state (the first token in the sequence).
        # This is often used as a summary representation of the entire sequence.
        hidden_state = output_1[0] #[batch_size, seq_len, hidden_size]
        pooler = hidden_state[:, 0] #[batch_size, hidden_size]

        # Pass through fully connected layer
        pooler = self.pre_classifier(pooler)
        # Apply ReLU activation to introduce non-linearity.
        pooler = torch.nn.ReLU()(pooler)
        # Apply dropout for regularization to reduce overfitting.
        pooler = self.dropout(pooler)#[batch_size, hidden_size]

        # Pass the result through the final classifier layer to get logits.
        # Maps the hidden size to the number of output classes ie.2
        output = self.classifier(pooler) #[batch_size, num_classes]
        return output
    
if __name__ == "__main__":
    pass