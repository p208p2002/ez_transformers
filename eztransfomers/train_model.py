# train model
from .core import computeAccuracy,log,blockPrint
class TrainManager():
    def __init__(self,
            model,
            optimizer,
            device = 'cpu', # cpu or cuda
            epoch=3,
            learning_rate=5e-6,
            log_interval = 50
        ):
        blockPrint()
        model.to(device)
        model.zero_grad()
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.epoch = epoch
        self.running_train_acc = 0.0
        self.running_train_loss = 0.0
        self.running_test_acc = 0.0
        self.running_test_loss = 0.0
        self.log_interval = log_interval
    
    def train(self,train_dataloader,test_dataloader = None):
        device = self.device
        optimizer = self.optimizer
        model = self.model
        total_epoch = self.epoch
        try:
            for epoch in range(total_epoch):
                # train
                self.running_train_loss = 0.0
                self.running_train_acc = 0.0
                for batch_index, batch_dict in enumerate(train_dataloader):
                    model.train()
                    batch_dict = tuple(t.to(device) for t in batch_dict)
                    outputs = model(batch_dict[0], labels=batch_dict[1])
                    loss, logits = outputs[:2]
                    loss.sum().backward()
                    optimizer.step()
                    model.zero_grad()

                    # compute the loss
                    loss_t = loss.item()
                    self.running_train_loss += (loss_t - self.running_train_loss) / (batch_index + 1)

                    # compute the accuracy
                    acc_t = computeAccuracy(logits, batch_dict[1])
                    self.running_train_acc += (acc_t - self.running_train_acc) / (batch_index + 1)

                    # log
                    if(batch_index % self.log_interval == 0):
                        log(">> TRAIN << epoch:%2d batch:%4d loss:%2.4f acc:%3.4f"%(epoch+1, batch_index+1, self.running_train_loss, self.running_train_acc))     
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        
            
                
