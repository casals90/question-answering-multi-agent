# Run / Stop docker container.
deploy_jupyter:
	docker compose -f docker-compose.yml up --build

jupyter_up:
	docker compose -f docker-compose.yml up

jupyter_down:
	docker compose -f docker-compose.yml down
