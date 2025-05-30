import torch
import json
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
import os

from .score import run_ner, process_embedding
from .utils import compute


DEFAULT_MATRIX_LONG = {"abnormality_abnormality": 0.4276119164393705, "abnormality_anatomy": 0.6240929990607657, "abnormality_disease": 0.0034478181112993847, "abnormality_non-abnormality": 0.5431049700217344, "abnormality_non-disease": 0.27005425386213877, "anatomy_abnormality": 0.7487824274337533, "anatomy_anatomy": 0.2856134859160784, "anatomy_disease": 0.4592143222158069, "anatomy_non-abnormality": 0.02097055139911715, "anatomy_non-disease": 0.00013736314126696204, "disease_abnormality": 0.8396510075734789, "disease_anatomy": 0.9950209388542061, "disease_disease": 0.8460555030578727, "disease_non-abnormality": 0.9820689020512646, "disease_non-disease": 0.3789136708096537, "non-abnormality_abnormality": 0.16546764653692908, "non-abnormality_anatomy": 0.018670610691852826, "non-abnormality_disease": 0.719397354576018, "non-abnormality_non-abnormality": 0.0009357166071730684, "non-abnormality_non-disease": 0.0927333564267591, "non-disease_abnormality": 0.7759420231214385, "non-disease_anatomy": 0.1839139293714062, "non-disease_disease": 0.10073046076318157, "non-disease_non-abnormality": 0.03860183811876373, "non-disease_non-disease": 0.34065681486566446, "neg_weight":0.8716553966489615}
DEFAULT_MATRIX_SHORT = {"abnormality_abnormality": 0.4070293318365468, "abnormality_anatomy": 0.6952639610605605, "abnormality_disease": 0.28342529466226446, "abnormality_non-abnormality": 0.9479148658006686, "abnormality_non-disease": 0.23875064111146294, "anatomy_abnormality": 0.5829759950441763, "anatomy_anatomy": 0.7709590751917746, "anatomy_disease": 0.0006059634829551632, "anatomy_non-abnormality": 0.794672584951181, "anatomy_non-disease": 0.27982942400798977, "disease_abnormality": 0.8840397619834857, "disease_anatomy": 0.9637659445696822, "disease_disease": 0.19018958438059513, "disease_non-abnormality": 0.6962283914800402, "disease_non-disease": 0.943727057946997, "non-abnormality_abnormality": 0.1712744286898638, "non-abnormality_anatomy": 0.4485149671497294, "non-abnormality_disease": 0.00045065329822896076, "non-abnormality_non-abnormality": 0.0007887930317199857, "non-abnormality_non-disease": 0.8555432840895761, "non-disease_abnormality": 0.9555801066212176, "non-disease_anatomy": 0.13122106162635216, "non-disease_disease": 0.6072996585919443, "non-disease_non-abnormality": 0.05650711141169969, "non-disease_non-disease": 0.3214769399791204, "neg_weight":0.3611577852354489}


class RaTEScore:
    def __init__(self, 
                    bert_model="Angelakeke/RaTE-NER-Deberta",
                    eval_model='FremyCompany/BioLORD-2023-C',
                    batch_size=1,
                    use_gpu=True,
                    visualization_path=None,
                    affinity_matrix="long",
                ):
        """ RaTEScore is a novel, entity-aware metric to assess the quality of medical reports generated by AI models. 
        It emphasizes crucial medical entities such as diagnostic outcomes and anatomical details, and is robust 
        against complex medical synonyms and sensitive to negation expressions. The evaluations demonstrate that 
        RaTEScore aligns more closely with human preference than existing metrics.

        Args:
            bert_model (str, optional): Medical entity recognition modul module. Defaults to "Angelakeke/RaTE-NER-Deberta".
            eval_model (str, optional): Synonym disambuation encoding module. Defaults to 'FremyCompany/BioLORD-2023-C'.
            batch_size (int, optional): Batch size to choose. Defaults to 1.
            use_gpu (bool, optional): If to use gpu. Defaults to True.
            visualization_path (str, optional): Output the visualized files, default to save as a json file. Defaults to None.
            affinity_matrix (str, optional):pre-searched type weight and can be changed due to the human rating bias. 
                                          Defaults to 'long'.

        """
        
        # if use_gpu
        if use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # load the Medical entity recognition module
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.model = AutoModelForTokenClassification.from_pretrained(bert_model).eval().to(self.device)
        
        # load the Synonym disambuation module
        self.eval_tokenizer = AutoTokenizer.from_pretrained(eval_model)
        self.eval_model = AutoModel.from_pretrained(eval_model).eval().to(self.device)
       
        # load the weight matrix
        if isinstance(affinity_matrix, str):
            # Choose the appropriate matrix based on the argument
            if affinity_matrix.lower() == "long":
                self.matrix_path = DEFAULT_MATRIX_LONG
            elif affinity_matrix.lower() == "short":
                self.matrix_path = DEFAULT_MATRIX_SHORT
            else:
                # Assume it's a file path
                try:
                    with open(affinity_matrix, 'r') as f:
                        self.matrix_path = json.load(f)
                except Exception as e:
                    raise ValueError(f"Failed to load affinity matrix from {affinity_matrix}: {e}")
        else:
            raise ValueError("affinity_matrix must be a string")
            
        self.affinity_matrix = {(k.split('_')[0].upper(), k.split('_')[1].upper()):v for k,v in self.matrix_path.items()}
        
        # load the label file
        self.config = AutoConfig.from_pretrained(bert_model)
        self.label2idx = self.config.label2id
        self.idx2label = self.config.id2label
        
        # save the input
        self.batch_size = batch_size
        
        if visualization_path:
            self.visualization_path = visualization_path
            if not os.path.exists(os.path.dirname(visualization_path)):
                os.makedirs(os.path.dirname(visualization_path))
        else:
            self.visualization_path = None
        
        
    def compute_score(self, candidate_list, reference_list):
        '''Compute the RaTEScore for the candidate and reference reports.
    
        Args:
            candidate_list (list): list of candidate reports
            reference_list (list): list of reference reports
        '''
        
        # check if candidate and reference are list
        if not isinstance(candidate_list, list):
            raise ValueError("candidate must be a list")
        if not isinstance(reference_list, list):
            raise ValueError("reference must be a list")
        
        assert len(candidate_list) == len(reference_list), "candidate and reference must have the same length"
        
        # check if candidate and reference are list of strings
        if not all(isinstance(x, str) for x in candidate_list):
            raise ValueError("candidate must be a list of strings")
        
        gt_pairs = run_ner(reference_list, self.idx2label, self.tokenizer, self.model, self.device, self.batch_size)
        pred_pairs = run_ner(candidate_list, self.idx2label, self.tokenizer, self.model, self.device, self.batch_size)
        
        rate_score = []
        
        for gt_pair, pred_pair in zip(gt_pairs, pred_pairs):
            
            # process the embedding for gt
            gt_embeds_word, gt_types = process_embedding(gt_pair, self.eval_tokenizer, self.eval_model, self.device)

            # process the embedding for pred
            pred_embeds_word, pred_types = process_embedding(pred_pair, self.eval_tokenizer, self.eval_model, self.device)
            
            # compute the score, if the length of gt or pred is 0, the score is 0.5
            if len(gt_embeds_word) == 0 or len(pred_embeds_word) == 0:
                rate_score.append(0.5)
                continue 

            precision_score = compute(gt_embeds_word, pred_embeds_word, gt_types, pred_types, self.affinity_matrix)
            recall_score = compute(pred_embeds_word, gt_embeds_word, pred_types, gt_types, self.affinity_matrix)

            if precision_score + recall_score == 0:
                rate_score.append(0)
            else:  
                rate_score.append(2*precision_score*recall_score/(precision_score+recall_score))
            
        if self.visualization_path:
            save_file = pd.DataFrame({
                'candidate': candidate_list,
                'reference': reference_list,
                'candidate_entities': pred_pairs,
                'reference_entities': gt_pairs,
                'rate_score': rate_score
            })
            save_file.to_json(os.path.join(self.visualization_path, 'rate_score.json'), lines=True, orient='records')
                
        return rate_score