from catalyst.dl import Runner
import torch


class CustomRunner(Runner):
    def _handle_batch(self, batch):
        seq, cont, y = batch
        pred = self.model(seq, cont)
        loss = self.criterion(pred, y)
        self.batch_metrics = {'loss': loss}
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    @torch.no_grad()
    def predict_batch(self, batch):
        batch = self._batch2device(batch, self.device)
        if len(batch) == 2:
            seq, cont = batch
        elif len(batch) == 3:
            seq, cont, _ = batch
        else:
            raise RuntimeError
        seq = seq.reshape(seq.shape[0], -1, seq.shape[1])
        pred = self.model(seq, cont)
        return pred
