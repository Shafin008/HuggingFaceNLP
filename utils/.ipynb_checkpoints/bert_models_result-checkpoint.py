import torch

models = ['bert', 'finbert'] 

def result_output_finbert(model, output_probs):
    results = []
    
    for prob_list in output_probs:
        prob, ix = torch.max(prob_list, dim=-1)

        if ix == torch.tensor(model.config.label2id['positive']):
            pos = {'label': 'POSITIVE', 'score': prob.item()}
            results.append(pos)

        elif ix == torch.tensor(model.config.label2id['negative']):
            neg = {'label': 'NEGATIVE', 'score': prob.item()}
            results.append(neg)

        elif ix == torch.tensor(model.config.label2id['neutral']):
            neut = {'label': 'NEUTRAL', 'score': prob.item()}
            results.append(neut)

    return results

def result_output_bert(model, output_probs):
    results = []
    
    for prob_list in output_probs:
        prob, ix = torch.max(prob_list, dim=-1)
    
        if ix == torch.tensor(model.config.label2id['POSITIVE']):
            pos = {'label': 'POSITIVE', 'score': prob.item()}
            results.append(pos)
    
        elif ix == torch.tensor(model.config.label2id['NEGATIVE']):
            neg = {'label': 'NEGATIVE', 'score': prob.item()}
            results.append(neg)

    return results

if __name__ == '__main__':
    for model in models:
        if model == 'finbert':
            result_output_finbert(model, output_probs)

        elif model == 'bert':
            result_output_bert(model, output_probs)