FROM ultralytics/ultralytics:latest-cpu

# Copy the project folder
COPY . /ultralytics/people_counter

# Set environment variable to make apt-get non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies from packages.txt
RUN apt-get update && \
    apt-get install ntp -y && \
    echo "System packages installed successfully!"

# Set the timezone to Asia/Colombo
RUN ln -fs /usr/share/zoneinfo/Asia/Colombo /etc/localtime && \
    echo "Asia/Colombo" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    echo "Timezone set to Asia/Colombo!"

# Set the working directory
WORKDIR /ultralytics/people_counter

# Start the Python scripts and write logs in append mode
CMD python3 main.py




# docker build -t people_counting:bootcamp .
# docker run -dt --name jetson -v D:\Senzmate_Cohort4\week04\people_counter_assignment\people_counter:/home/people_counter  --restart always people_counting:bootcamp
# docker exec -it peoplecounter /bin/bash
# docker logs -f jetson
# docker stop jetson

