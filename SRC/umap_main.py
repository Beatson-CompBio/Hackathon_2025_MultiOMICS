import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from machine_learning.base_classes.early_and_single import Early_or_Single_Model



def get_umap(train_data, val_data, n_components=2, random_state=42, **kwargs):
    reducer = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
    train_embedding = reducer.fit_transform(train_data)
    val_embedding = reducer.transform(val_data)
    return train_embedding, val_embedding, reducer


def plot_umap(
    train_emb,
    val_emb,
    train_y,
    val_y,
    show_val=True,
    figsize=(8, 6),
    alpha=0.7,
    train_alpha_factor=0.3,
    point_size=50,
    train_marker='o',
    val_marker='*',
    palette=None,
    class_labels=None
):
    import matplotlib.lines as mlines
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # 1) Build DataFrames
    df_train = pd.DataFrame(train_emb, columns=['UMAP1', 'UMAP2'])
    df_train['Class'] = train_y
    df_train['Dataset'] = 'Train'

    df_val = pd.DataFrame(val_emb, columns=['UMAP1', 'UMAP2'])
    df_val['Class'] = val_y
    df_val['Dataset'] = 'Validation'

    # 2) Gather all unique class labels (train first, then any new from val)
    all_classes = list(dict.fromkeys(
        list(df_train['Class'].unique()) + list(df_val['Class'].unique())
    ))

    # 3) Build a class → color mapping
    if palette is None:
        color_list = sns.color_palette(n_colors=len(all_classes))
        class_to_color = {cls: color_list[i] for i, cls in enumerate(all_classes)}
    else:
        if isinstance(palette, dict):
            class_to_color = palette
        else:
            expanded = sns.color_palette(palette, n_colors=len(all_classes))
            class_to_color = {cls: expanded[i] for i, cls in enumerate(all_classes)}

    # 4) Plot train points
    train_alpha = alpha * train_alpha_factor if show_val else alpha
    sns.scatterplot(
        data=df_train,
        x='UMAP1', y='UMAP2',
        hue='Class',
        palette=class_to_color,
        alpha=train_alpha,
        s=point_size,
        marker=train_marker,
        ax=ax,
        legend=False
    )

    # 5) Plot validation points if requested
    if show_val:
        sns.scatterplot(
            data=df_val,
            x='UMAP1', y='UMAP2',
            hue='Class',
            palette=class_to_color,
            alpha=alpha,
            s=point_size * 1.2,
            marker=val_marker,
            ax=ax,
            legend=False
        )

    ax.set_title('UMAP Projection{}'.format(' (Validation ★)' if show_val else ''))
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')

    # 6) Manually build the “Class” legend (always)
    class_handles = [
        mlines.Line2D(
            [], [],
            color=class_to_color[cls],
            marker='o',
            linestyle='None',
            markersize=8,
            label=f'{class_labels[cls]}' if class_labels and cls in class_labels else f'Class {cls}'
        )
        for cls in all_classes
    ]
    class_legend = ax.legend(
        handles=class_handles,
        title='Class',
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0.
    )
    ax.add_artist(class_legend)

    # 7) Only add “Dataset” legend if show_val=True
    train_handle = mlines.Line2D(
        [], [],
        color='gray',
        marker=train_marker,
        linestyle='None',
        markersize=8,
        label='Train'
    )
    handles = [train_handle]
    if show_val:
        val_handle = mlines.Line2D(
            [], [],
            color='gray',
            marker=val_marker,
            linestyle='None',
            markersize=10,
            label='Validation'
        )
        handles.append(val_handle)
    ax.legend(
        handles=handles,
        title='Dataset',
        loc='upper left',
        bbox_to_anchor=(1.01, 0.75),
        borderaxespad=0.
    )

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # leave space for legends
    return fig, ax



class_labels = {0: "ductal", 1: "lobular"}

modalities = ['hist', 'cnv', 'rna', 'meth', 'mir', 'mutation']

for modality in modalities:
    print(f"Processing modality: {modality}")
    train_modalities = {
        'cnv': pd.read_csv(f"../processed_data/train_{modality}.csv"),
    }

    val_modalities = {
        'cnv': pd.read_csv(f"../processed_data/val_{modality}.csv"),
    }
    model = Early_or_Single_Model()
    train_x, train_y = model.data(train_modalities)
    val_x, val_y = model.data(val_modalities)

    train_x = train_x.drop(columns=['Unnamed: 0'])
    val_x = val_x.drop(columns=['Unnamed: 0'])


    train_emb, val_emb, umap_reducer = get_umap(train_x, val_x, n_components=2)


    fig1, ax1 = plot_umap(
        train_emb, val_emb,
        train_y, val_y,
        show_val=True,
        palette=None,
        class_labels=class_labels,
    )
    plt.title(
        f"UMAP Projection with Validation Data ★ : "
        + r"$\bf{" + modality.upper() + "}$"
        + f"  : {train_x.shape[1]} features"
    )
    fig1.savefig(f"./streamlit_app/umaps/{modality}_umap_with_val.png", dpi=200)
    plt.close(fig1)

    fig2, ax2 = plot_umap(
        train_emb, val_emb,
        train_y, val_y,
        show_val=False,
        palette=None,
        class_labels=class_labels,
    )
    plt.title(
        f"UMAP Projection : "
        + r"$\bf{" + modality.upper() + "}$"
        + f"  : {train_x.shape[1]} features"
    )
    fig2.savefig(f"./streamlit_app/umaps/{modality}_umap.png", dpi=200)
    plt.close(fig2)