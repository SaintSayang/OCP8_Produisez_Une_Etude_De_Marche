import os
import warnings

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from tabulate import tabulate
from matplotlib.colors import ListedColormap


def analyse_univariee(df):
    """
    Effectue une analyse univariée sur toutes les variables d'un DataFrame.

    Pour les variables numériques, des statistiques descriptives, un histogramme et un boxplot sont générés.
    Pour les variables non numériques, seules des statistiques descriptives sont générées.

    Paramètres
    ----------
    df : DataFrame
        DataFrame pandas à analyser.
    """

    for col in df.columns:
        # Affichage du nom de la variable analysée
        print(f"Analyse univariée pour la variable '{col}':")
        # Vérifie si la colonne est numérique
        if np.issubdtype(df[col].dtype, np.number):
            # Si oui, calcul des statistiques descriptives
            analyse = df[col].describe().to_frame().transpose()
            analyse["skew"] = df[col].skew()
            analyse["kurtosis"] = df[col].kurt()
        else:
            # Si non, calcul des statistiques de base pour les variables catégorielles
            analyse = df[col].describe(include=["O"]).to_frame().transpose()

        # Affichage du résultats sous forme de tableau
        print(tabulate(analyse, headers="keys", tablefmt="fancy_grid"))

        # Vérifie si la colonne est numérique
        if np.issubdtype(df[col].dtype, np.number):
            # Si oui, création d'une figure pour tracer les graphiques
            plt.figure(figsize=(7, 4))

            # Trace un histogramme avec une courbe de densité de probabilité
            ax1 = sns.histplot(data=df, x=col, kde=True)
            ax1.set(title=f"Analyse univariée de la variable {col}")
            if ax1.lines:
                ax1.lines[0].set(color="orange", linewidth=3)
            # Ajoute des lignes verticales pour la moyenne et la médiane
            ax1.axvline(
                df[col].mean(),
                color="crimson",
                linestyle="dotted",
                label=f"Moyenne {col}",
            )
            plt.axvline(
                df[col].median(),
                color="black",
                linestyle="dotted",
                label=f"Médiane {col}",
            )
            plt.legend()
            plt.show()

            # Création d'une autre figure pour le boxplot
            plt.figure(figsize=(7, 4))

            # Trace un boxplot pour visualiser la répartition des données
            ax2 = sns.boxplot(
                x=col, data=df, linewidth=3, color="white", showmeans=True
            )
            # Donner un titre dynamique au boxplot
            ax2.set(title=f"Répartition de la variable {col}")
            plt.show()

        # Insértion d'une ligne vide pour séparer les résultats de chaque variable
        print("\n")


def comp_analyse_univariee(df1, df2):
    """
    Effectue une analyse univariée sur toutes les variables de deux DataFrames.

    Pour les variables numériques, des statistiques descriptives, un histogramme et un boxplot sont générés.
    Pour les variables non numériques, seules des statistiques descriptives sont générées.

    Paramètres
    ----------
    df1, df2 : DataFrame
        DataFrames pandas à analyser.
    """

    for col in df1.columns:
        # Assurer que la colonne existe dans les deux DataFrames
        if col not in df2.columns:
            continue

        # Affichage du nom de la variable analysée
        print(f"Analyse univariée pour la variable '{col}':")

        # Vérifie si la colonne est numérique
        if np.issubdtype(df1[col].dtype, np.number) and np.issubdtype(
            df2[col].dtype, np.number
        ):
            # Si oui, calcul des statistiques descriptives
            analyse1 = df1[col].describe().to_frame().transpose()
            analyse1["skew"] = df1[col].skew()
            analyse1["kurtosis"] = df1[col].kurt()

            analyse2 = df2[col].describe().to_frame().transpose()
            analyse2["skew"] = df2[col].skew()
            analyse2["kurtosis"] = df2[col].kurt()

        else:
            # Si non, calcul des statistiques de base pour les variables catégorielles
            analyse1 = df1[col].describe(include=["O"]).to_frame().transpose()
            analyse2 = df2[col].describe(include=["O"]).to_frame().transpose()

        # Affichage du résultats sous forme de tableau
        print("DataFrame 1")
        print(tabulate(analyse1, headers="keys", tablefmt="fancy_grid"))
        print("DataFrame 2")
        print(tabulate(analyse2, headers="keys", tablefmt="fancy_grid"))

        # Vérifie si la colonne est numérique
        if np.issubdtype(df1[col].dtype, np.number) and np.issubdtype(
            df2[col].dtype, np.number
        ):
            # Si oui, création d'une figure pour tracer les graphiques
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            # Trace un histogramme avec une courbe de densité de probabilité
            sns.histplot(data=df1, x=col, kde=True, ax=axes[0])
            axes[0].set(title=f"Analyse univariée de la variable {col} (DataFrame 1)")

            sns.histplot(data=df2, x=col, kde=True, ax=axes[1])
            axes[1].set(title=f"Analyse univariée de la variable {col} (DataFrame 2)")

            plt.show()

            # Création d'une autre figure pour le boxplot
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            # Trace un boxplot pour visualiser la répartition des données
            sns.boxplot(
                x=col, data=df1, linewidth=3, color="white", showmeans=True, ax=axes[0]
            )
            axes[0].set(title=f"Répartition de la variable {col} (DataFrame 1)")

            sns.boxplot(
                x=col, data=df2, linewidth=3, color="white", showmeans=True, ax=axes[1]
            )
            axes[1].set(title=f"Répartition de la variable {col} (DataFrame 2)")

            plt.show()

        # Insértion d'une ligne vide pour séparer les résultats de chaque variable
        print("\n")





def projection_acp(scores_pca, n_components, labels):
    """
    Affiche la projection des données sur les plans factoriels générés par l'ACP.

    Arguments :
    scores_pca -- Tableau numpy des coordonnées des données projetées
    n_components -- Nombre de composantes principales
    labels -- Liste des labels ou noms des points à afficher

    """

    # Création d'un DataFrame avec les coordonnées
    df_scores = pd.DataFrame(
        scores_pca, columns=["F" + str(i + 1) for i in range(scores_pca.shape[1])]
    )

    # Plot sur les plans factoriels
    for i in range(0, n_components - 1, 2):
        plt.figure(figsize=(10, 7))
        plt.scatter(df_scores["F" + str(i + 1)], df_scores["F" + str(i + 2)])
        for j, txt in enumerate(labels):
            plt.annotate(
                txt, (df_scores["F" + str(i + 1)][j], df_scores["F" + str(i + 2)][j])
            )
        plt.xlabel("F" + str(i + 1))
        plt.ylabel("F" + str(i + 2))
        plt.title("Projection sur le plan factoriel F" + str(i + 1) + "/F" + str(i + 2))
        plt.show()


def generation_boxplots(df, x_var):
    """
    Cette fonction génère des boxplots pour chaque colonne numérique
    dans le DataFrame (sauf pour celles contenant le mot "cluster"),
    avec la variable `x` spécifiée sur l'axe des x. Les boxplots sont
    disposés en subplots avec deux plots par ligne.

    Arguments:
    df -- DataFrame contenant les données à tracer
    x_var -- Nom de la colonne à utiliser pour l'axe des x
    """
    num_plots = sum("cluster" not in col for col in df.columns)
    num_rows = (num_plots + 1) // 2

    fig, axs = plt.subplots(num_rows, 2, figsize=(20, 6 * num_rows))

    # Aplatir la liste de Axes pour une itération facile
    axs = axs.flatten()

    # Index pour le subplot actuel
    ax_idx = 0

    for col in df.columns:
        if "cluster" in col:  # Pas de boxplot pour les colonnes contenant 'cluster'
            continue

        sns.boxplot(x=x_var, y=col, data=df, ax=axs[ax_idx])

        # La moyenne totale pour la variable courante est calculée
        moyenne_totale = df[col].mean()

        # Une ligne horizontale pointillée rouge est ajoutée pour la moyenne totale
        axs[ax_idx].axhline(y=moyenne_totale, color="r", linestyle="--")

        axs[ax_idx].set_title(f"Boxplot de {col} pour chaque cluster")

        ax_idx += 1

    # Supprimer les subplots inutilisés
    for i in range(ax_idx, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()


def projection_acp_clustering(scores_pca, n_components, clusters, labels=None):
    """
    Affiche la projection des données et des centroides sur les plans factoriels générés par l'ACP.
    Les individus sont colorés en fonction des clusters.

    Arguments :
    scores_pca -- Tableau numpy des coordonnées des données projetées
    n_components -- Nombre de composantes principales
    clusters -- Tableau ou série indiquant les clusters correspondants aux individus
    labels -- Liste des labels ou noms des points à afficher (optionnel)

    """

    # Création d'un DataFrame avec les coordonnées
    df_scores = pd.DataFrame(
        scores_pca, columns=["F" + str(i + 1) for i in range(scores_pca.shape[1])]
    )

    # Ajout des labels si fournis
    if labels is not None:
        df_scores["Labels"] = labels

    # Ajout des clusters au DataFrame des scores
    df_scores["Cluster"] = clusters

    # Calcul des centroides de chaque cluster
    centroids = df_scores.groupby("Cluster").mean()

    # Plot sur les plans factoriels
    for i in range(0, n_components - 1, 2):
        plt.figure(figsize=(10, 7))
        plt.scatter(
            df_scores["F" + str(i + 1)],
            df_scores["F" + str(i + 2)],
            c=df_scores["Cluster"],
            cmap="coolwarm",
        )
        plt.scatter(
            centroids["F" + str(i + 1)],
            centroids["F" + str(i + 2)],
            marker="x",
            s=200,
            linewidths=3,
            color="r",
        )
        if labels is not None:
            for j, txt in enumerate(labels):
                plt.annotate(
                    txt,
                    (df_scores["F" + str(i + 1)][j], df_scores["F" + str(i + 2)][j]),
                )
        plt.xlabel("F" + str(i + 1))
        plt.ylabel("F" + str(i + 2))
        plt.title("Projection sur le plan factoriel F" + str(i + 1) + "/F" + str(i + 2))
        plt.show()


def radar_clustering(df, cluster_col):
    """
    Crée des graphiques radar (radar plots) pour chaque cluster dans un dataframe.

    Arguments:
    df -- DataFrame contenant les données des clusters et des variables
    cluster_col -- Nom de la colonne contenant les informations de clustering

    """

    # Sélectionner uniquement les colonnes pertinentes pour le graphique radar
    cols_to_keep = [
        col for col in df.columns if "cluster" not in col or col == cluster_col
    ]
    df = df[cols_to_keep]

    # Calculer les moyennes pour chaque cluster et chaque variable
    moyennes_clusters = df.groupby(cluster_col).mean()

    # Normaliser les moyennes pour chaque variable
    moyennes_clusters_norm = (moyennes_clusters - moyennes_clusters.min()) / (
        moyennes_clusters.max() - moyennes_clusters.min()
    )

    # Créer un radar plot pour chaque cluster
    num_clusters = moyennes_clusters_norm.shape[0]
    num_rows = 4
    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 20), subplot_kw={"polar": True})

    # Supprimer le dernier subplot qui est inutile
    fig.delaxes(axs[-1, -1])

    # Nombre de variables
    num_vars = len(moyennes_clusters_norm.columns)

    # Calculer l'angle de chaque axe
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Les axes doivent revenir au début pour fermer le graphique radar
    angles += angles[:1]

    # Utiliser une palette de couleurs pour donner une couleur différente à chaque subplot
    colors = plt.cm.get_cmap("hsv", num_clusters)

    for idx, (i, row) in enumerate(moyennes_clusters_norm.iterrows()):
        ax = axs[idx // 2, idx % 2]
        ax.set_thetagrids(np.degrees(angles[:-1]), moyennes_clusters_norm.columns)
        values = row.values.flatten().tolist()
        values += values[:1]  # Répéter la première valeur pour fermer le graphique
        ax.plot(angles, values, linewidth=1, color=colors(idx), label=f"Cluster {i}")
        ax.fill(angles, values, color=colors(idx), alpha=0.25)
        ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.show()


# Créer une colormap discrète personnalisée
def discrete_cmap(n, base_cmap=None):
    """Crée une colormap discrète avec n couleurs."""
    if base_cmap is None:
        base_cmap = plt.cm.get_cmap('Dark2')
    colors = base_cmap(np.linspace(0, 1, n))
    cmap = ListedColormap(colors)
    return cmap



def detecter_valeurs_aberrantes(df):
    """
    Détecte les valeurs aberrantes dans toutes les colonnes numériques d'un DataFrame en utilisant la méthode de l'IQR.

    Paramètres
    ----------
    df : DataFrame
        DataFrame à analyser.
    """
    for col in df.columns:
        # Vérifier si la colonne est numérique
        if np.issubdtype(df[col].dtype, np.number):
            # Calculer l'IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Définir les limites
            limite_inf = Q1 - 1.5 * IQR
            limite_sup = Q3 + 1.5 * IQR

            # Identifier les valeurs aberrantes
            valeurs_aberrantes = df[(df[col] < limite_inf) | (df[col] > limite_sup)][col]

            # Afficher le tableau des valeurs aberrantes
            if not valeurs_aberrantes.empty:
                print(f"Valeurs aberrantes pour la variable '{col}' :")
                print(tabulate(valeurs_aberrantes.to_frame(), headers='keys', tablefmt='fancy_grid'))
                print("\n")
            else:
                print(f"Aucune valeur aberrante détectée pour la variable '{col}'.\n")
        else:
            print(f"La variable '{col}' n'est pas numérique, elle est ignorée.\n")
            
            
def filtre_clusters(df, cluster_variable, clusters):
    """
    Filtre un DataFrame pour sélectionner les pays des clusters spécifiés pour une méthode de clustering donnée.

    Arguments :
    df : DataFrame
        DataFrame contenant les données avec les pays en index.
    cluster_variable : str
        Nom de la variable de clustering.
    clusters : list
        Liste des clusters à sélectionner.

    Returns :
    DataFrame
        Nouveau DataFrame contenant les pays des clusters spécifiés avec les colonnes 'pays' et 'cluster'.
    """
    df_choix = df[df[cluster_variable].isin(clusters)]
    df_choix = df_choix.reset_index()
    df_choix = df_choix[['pays', cluster_variable]]
    return df_choix


def biplot(score, coeff, pc1, pc2, labels=None):
    """
    Affiche un biplot, qui est une représentation graphique des scores des individus 
    et des poids des variables pour deux composantes principales spécifiques.

    Paramètres:
    - score : tableau numpy contenant les scores des individus.
    - coeff : tableau numpy contenant les poids des variables.
    - pc1 : int, indice de la première composante principale à afficher.
    - pc2 : int, indice de la deuxième composante principale à afficher.
    - labels : list, noms des variables. Si None, les variables sont nommées Var1, Var2, etc.

    Retourne:
    - Un graphique matplotlib.
    """
    # Extraction des scores pour les deux composantes principales
    xs = score[:,pc1]
    ys = score[:,pc2]
    n = coeff.shape[0]

    # Calcul des facteurs d'échelle pour normaliser les données
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())

    # Création du nuage de points pour les individus
    plt.scatter(xs * scalex, ys * scaley)

    # Ajout des vecteurs pour les variables
    for i in range(n):
        plt.arrow(0, 0, coeff[i,pc1], coeff[i,pc2],color='r',alpha=0.5)
        if labels is None:
            plt.text(coeff[i,pc1]* 1.15, coeff[i,pc2] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i,pc1]* 1.15, coeff[i,pc2] * 1.15, labels[i], color='g', ha='center', va='center')

    # Configuration des limites et des labels des axes
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(pc1+1))
    plt.ylabel("PC{}".format(pc2+1))

    # Ajout d'une grille
    plt.grid()
