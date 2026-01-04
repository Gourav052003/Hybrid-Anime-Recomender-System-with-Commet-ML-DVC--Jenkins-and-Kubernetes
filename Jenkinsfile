pipeline{
    agent any

    stages{
        stage('cloning from github'){
            steps{
                script{
                    echo 'cloning the repo'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'GitHub-Token', url: 'https://github.com/Gourav052003/Hybrid-Anime-Recomender-System-with-Commet-ML-DVC--Jenkins-and-Kubernetes.git']])
                }
            }
        }
    }
}