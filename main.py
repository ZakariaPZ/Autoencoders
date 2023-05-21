from torchvision import datasets, transforms
from torch.utils import data
import torch
import argparse
import os
import tqdm
from autoencoders import BetaVAE

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")


def main():
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, 
                                transform=transforms.ToTensor())

    batch_size = 64
    dataloader = data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

    model = BetaVAE().to(device)
    loss = VAELoss()

    learning_rate = 1e-4
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 30

    for epoch in range(epochs):

        for i, train_data in enumerate(tqdm(dataloader)):
            x, _ = train_data

            x = x.to(device)
            xtilde, mu, log_variance = model(x)
            error = loss(xtilde, x, mu, log_variance)

            optim.zero_grad()
            error.backward()
            optim.step()

    torch.save(model, 'models/small_VAE.pt')

if __name__ == '__main__':
    main()

def main(args):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])
    test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = data.DataLoader(train_set, batch_size=args.bs, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=args.bs, shuffle=False)
    test_loader = data.DataLoader(test_set, batch_size=args.bs, shuffle=False)



    # pl.seed_everything(42)
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])
    # test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # train_loader = data.DataLoader(train_set, batch_size=args.bs, shuffle=True)
    # val_loader = data.DataLoader(val_set, batch_size=args.bs, shuffle=False)
    # test_loader = data.DataLoader(test_set, batch_size=args.bs, shuffle=False)

    
    # if not os.path.exists(args.path):
    #     os.makedirs(args.path)
    
    # # save the run details and model
    # file_name = "\model.ckpt"
    # model_path = args.path + model_path

    # if args.type == 'CNN':
    #     model = ConvAutoEncoder()
    # else:
    #     model = LinearAutoEncoder()

    # if not os.path.isfile(model_path):

    #     trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[LossCallback()]) # add callbacks for sample reconstruction per epoch
    #     trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    #     with open(args.path + '/config.txt', 'w') as f:
    #         epochs = "Epochs: " + str(args.epochs) + '\n'
    #         bs = "Batch Size: " + str(args.bs) + '\n'
    #         model_type = args.type
    #         f.writelines([epochs, bs, model_type])

    #     checkpoint_path = model_path
    #     trainer.save_checkpoint(checkpoint_path)

    # else:
    #     model = model.load_from_checkpoint(model_path)

    # test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    # print('\nTest Error: ' + test_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='MNIST Autoencoders')
    parser.add_argument('--model_folder', type=str, required=True, help='Folder to save run details in.')
    parser.add_argument('--epochs', type=str, required=True, help='Number of epochs to train for.')
    parser.add_argument('--bs', type=str, required=True, help='Batch size.')
    parser.add_argument('--type', type=str, required=True, help='CNN or Linear.')

    args = parser.parse_args()
    main(args)