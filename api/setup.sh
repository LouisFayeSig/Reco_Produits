#!/usr/bin/env bash
set -e  # Arrête le script en cas d'erreur

# Définir le dossier du projet
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Nom de l'environnement virtuel
VENV_DIR=".venv-api"

echo "📦 Création et activation de l'environnement virtuel : $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"


# Installer les dépendances Python si requirements.txt existe
if [ -f "requirements-api.txt" ]; then
    echo "🐍 Installation des dépendances Python depuis requirements-api.txt..."
    pip install --upgrade pip
    pip install --no-cache-dir -r requirements-api.txt
else
    echo "⚠️ Aucun fichier requirements-api.txt trouvé."
fi

echo "✅ Installation terminée avec succès !"
