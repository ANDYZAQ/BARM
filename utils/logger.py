from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self, path) -> None:
        super(Logger, self).__init__()
        self.tbd_writer = SummaryWriter(path)

    def write_loss(self, loss, iter):
        self.tbd_writer.add_scalar('loss', loss, iter)

    def write_score(self, score, epoch):
        self.tbd_writer.add_scalar('miou', score, epoch)

    def __delattr__(self, __name: str) -> None:
        self.tbd_writer.close()