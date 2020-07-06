import matplotlib.pyplot as plt

def plot_history_regression(histories, key='mean_absolute_error', key2='loss'):
    plt.figure(figsize=(6, 6))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key], '--', label='MAE' + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),  label='MAE' + ' Train')

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key2],  '--', label='MSE' + ' Val')
        plt.plot(history.epoch, history.history[key2], color=val[0].get_color(),  label='MSE' + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])


def plot_history_classification(histories, key='acc', key2='loss'):
    plt.figure(figsize=(6, 8))

    plt.subplot(2, 1, 1)
    plt.grid(True)

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key2],  '--', label='Loss Val')
        plt.plot(history.epoch, history.history[key2], color=val[0].get_color(),  label='Loss Train')

    plt.xlabel('Epochs')
    plt.ylabel(key2.replace('_', ' ').title())
    plt.legend()

    plt.subplot(2, 1, 2)

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key],    '--', color='orange', label=name.title() + 'Acc Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),  label=name.title() + 'Acc Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])