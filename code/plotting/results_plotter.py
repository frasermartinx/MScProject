import matplotlib.pyplot as plt


def plotter(input,prediction,solution,X,Y):
    fig, axs = plt.subplots(nrows = 1, ncols = 3,gridspec_kw={'width_ratios': [1, 1, 1]})
    fig.set_figwidth(20)


    fig0 = axs[0].pcolormesh(X,Y,input.cpu().detach().numpy())
    fig.colorbar(fig0)
    axs[0].set_title("Input Function")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    #prediction
    fig1 = axs[1].pcolormesh(X,Y,prediction[0,0,:,:].cpu().detach().numpy())
    fig.colorbar(fig1)
    axs[1].set_title("Model Prediction")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")



    #actual
    #fig2 = axs[2].pcolormesh(X,Y,U.unsqueeze(0).unsqueeze(0))[0,0,:,:].cpu())
    fig2 = axs[2].pcolormesh(X,Y,solution.cpu())
    fig.colorbar(fig2)
    axs[2].set_title("Numerical Solution")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    plt.show()