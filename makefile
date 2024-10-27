# prepare documents and features for recommendations
run_preprocessing:
	cd preprocessing && python data_preparation.py

# run frontend and backend in docker container
run:
	docker-compose up -d --build

# stop containers
stop:
	docker-compose down

# run frontend for development
run_frontend:
	echo "Starting frontend"
	cd frontend && streamlit run frontend.py

# run backend for development
run_ranking:
	echo "Starting ranking service"
	cd ranking_service && python app.py

# retrain the scoring model for recommendations based on recent logs and features
run_training:
	python ./scoring_model/train_model.py

# docker logs
logs_frontend:
	docker logs -f article-recommender-system-frontend-1 --tail 100

logs_backend:
	docker logs -f article-recommender-system-ranking-service-1 --tail 100

