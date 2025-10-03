#!/usr/bin/env bash
set -e  # Arrête le script en cas d'erreur

# Définir le dossier du projet
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Nom de l'environnement virtuel
VENV_DIR=".venv-training"

echo "📦 Création et activation de l'environnement virtuel : $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Installer les dépendances système si apt.txt existe
if [ -f "apt.txt" ]; then
    echo "🔧 Installation des dépendances système depuis apt.txt..."
    sudo apt-get update
    xargs -a apt.txt sudo apt-get install -y
else
    echo "⚠️ Aucun fichier apt.txt trouvé, passage à l'étape suivante."
fi

# Installer les dépendances Python si requirements.txt existe
if [ -f "requirements-train.txt" ]; then
    echo "🐍 Installation des dépendances Python depuis requirements-train.txt..."
    pip install --upgrade pip
    pip install --no-cache-dir -r requirements-train.txt
else
    echo "⚠️ Aucun fichier requirements-train.txt trouvé."
fi

echo "✅ Installation terminée avec succès !"
