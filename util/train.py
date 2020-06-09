import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F


# 学習用関数の定義
def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src = batch.dial
        trg = batch.act

        optimizer.zero_grad()

#         output, _ = model(src, trg[:,:-1])
        output = model(src, trg)
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
        output_size = output.size()
        output_dim = output.shape[-1]

#         output = output.contiguous().view(-1, output_dim)
#         trg = trg[:,1:].contiguous().view(-1)
        ones = torch.sparse.torch.eye(5).to(device)
        trg=ones.index_select(0,trg).long()
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
        trg1 = trg.view(16,1,5)
        trg2 = torch.cat(([trg1]*output_size[1]),dim=1)
        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# 評価用関数の定義
def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch.dial
            trg = F.one_hot(batch.act)

            output = model(src, trg)

            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]

#             output_dim = output.shape[-1]

#             output = output.contiguous().view(-1, output_dim)
# #             trg = trg[:,1:].contiguous().view(-1)
#             ones = torch.sparse.torch.eye(5).to(device)
#             trg=ones.index_select(0,trg)
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]

            loss = criterion(output, trg)   

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# 処理時間測定用関数の定義
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 文章生成用関数の定義
def translate_sentence(sentence, src_field, trg_field, model, device, max_len=1000):

    model.eval()

    tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:,-1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention