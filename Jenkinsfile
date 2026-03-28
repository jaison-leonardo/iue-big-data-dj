pipeline {
    agent any

    environment {
        // Variables de entorno para Python
        PYTHONUNBUFFERED = '1'
    }

    stages {
        stage('Checkout del repositorio') {
            steps {
                // Extrae el código fuente
                checkout scm
            }
        }

        stage('Instalación de dependencias') {
            steps {
                echo 'Creando entorno virtual e instalando dependencias (para Docker Linux)...'
                // Se utiliza python3-venv para evitar colisiones con el sistema Debian interno
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Pruebas del Dataset') {
            steps {
                echo 'Verificando y probando los datos base...'
                // Ejecuta la suite con el python del entorno virtual
                sh '''
                    . venv/bin/activate
                    python -m unittest discover -s tests
                '''
            }
        }

        stage('Ejecución del pipeline principal') {
            steps {
                echo 'Corriendo el pipeline de Machine Learning...'
                sh '''
                    . venv/bin/activate
                    python pipeline/main_pipeline.py
                '''
            }
        }
    }

    post {
        always {
            echo 'Guardando artefactos (métricas, gráficas y modelos)...'
            archiveArtifacts artifacts: 'reports/figures/*.png, reports/metrics/*.json, models/trained/*.pkl', allowEmptyArchive: true
        }
        success {
            echo '¡Pipeline ejecutado con éxito!'
        }
        failure {
            echo 'Error en la ejecución. Revisa los logs de Jenkins.'
        }
    }
}
