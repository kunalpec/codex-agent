FROM maven:3.9-eclipse-temurin-21

ENV MAVEN_OPTS="-Dmaven.repo.local=/opt/packages/maven-repository" \
    GRADLE_USER_HOME=/opt/packages/gradle

RUN apt-get update \
    && apt-get install -y --no-install-recommends gradle \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["java", "--version"]
