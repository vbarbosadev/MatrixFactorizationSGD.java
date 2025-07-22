package SGD;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import com.google.gson.reflect.TypeToken;
import org.jetbrains.annotations.NotNull;

import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

public class MovieRecommenderSGD {


    private static class Rating {
        String user_id;
        String title;
        List<String> genres;
        double rating;

        public Rating(String user_id, String title, List<String> genres, double rating) {
            this.user_id = user_id;
            this.title = title;
            this.genres = genres;
            this.rating = rating;
        }

        public String getUserId() { return user_id; }
        public String getTitle() { return title; }
        public List<String> getGenres() { return genres; }
        public double getRating() { return rating; }
    }


    private static final int NUM_FEATURES = 10;
    private static final double LEARNING_RATE = 0.01;
    private static final double REGULARIZATION = 0.02;
    private static final int NUM_EPOCHS = 100;

    private static ConcurrentHashMap<String, double[]> userFactors = new ConcurrentHashMap<>();
    private static ConcurrentHashMap<String, double[]> genreFactors = new ConcurrentHashMap<>();
    private static Set<String> allUsers = ConcurrentHashMap.newKeySet();
    private static Set<String> allGenres = ConcurrentHashMap.newKeySet();

    private static final Gson gson = new Gson();
    private static final Type ratingListType = new TypeToken<List<Rating>>() {}.getType();

    public static ConcurrentLinkedQueue<Rating> loadRatingsParallel(Set<String> filenames) {
        ConcurrentLinkedQueue<Rating> ratings = new ConcurrentLinkedQueue<>();
        ExecutorService virtualThreadExecutor = Executors.newVirtualThreadPerTaskExecutor();

        CountDownLatch latch = new CountDownLatch(filenames.size());

        for (String filename : filenames) {
            Runnable task = () -> {
                try (FileReader reader = new FileReader(filename)) {
                    List<Rating> localList = gson.fromJson(reader, ratingListType);
                    if (localList != null) {
                        ratings.addAll(localList);
                    }
                } catch (IOException | JsonSyntaxException e) {
                    System.err.println("Erro ao processar arquivo: " + filename + " - " + e.getMessage());
                } finally {
                    latch.countDown();
                }
            };
            virtualThreadExecutor.execute(task);
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Thread principal interrompida enquanto aguardava o carregamento.");
        }

        virtualThreadExecutor.shutdown();
        return ratings;
    }

    public static void initializeFactors(@NotNull ConcurrentLinkedQueue<Rating> ratings) {
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        CountDownLatch extractionLatch = new CountDownLatch(ratings.size());
        for(Rating r : ratings) {
            executor.execute(() -> {
                allUsers.add(r.getUserId());
                allGenres.addAll(r.getGenres());
                extractionLatch.countDown();
            });
        }
        try {
            extractionLatch.await();
        } catch (InterruptedException e) { Thread.currentThread().interrupt(); }

        CountDownLatch initLatch = new CountDownLatch(allUsers.size() + allGenres.size());
        allUsers.forEach(user -> executor.execute(() -> {
            double[] features = ThreadLocalRandom.current().doubles(NUM_FEATURES, 0, 0.1).toArray();
            userFactors.put(user, features);
            initLatch.countDown();
        }));
        allGenres.forEach(genre -> executor.execute(() -> {
            double[] features = ThreadLocalRandom.current().doubles(NUM_FEATURES, 0, 0.1).toArray();
            genreFactors.put(genre, features);
            initLatch.countDown();
        }));

        try {
            initLatch.await();
        } catch (InterruptedException e) { Thread.currentThread().interrupt(); }

        executor.shutdown();
    }


    public static void trainModel(ConcurrentLinkedQueue<Rating> ratings) {
        System.out.println("Iniciando Treinamento com Executor e Runnable.");
        List<Rating> ratingList = new ArrayList<>(ratings);
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        int numThreads = Runtime.getRuntime().availableProcessors();
        int chunkSize = (int) Math.ceil((double) ratingList.size() / (numThreads * 4));
        if (chunkSize == 0 && !ratingList.isEmpty()) chunkSize = 1;

        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            List<Runnable> tasks = new ArrayList<>();
            for (int i = 0; i < ratingList.size(); i += chunkSize) {
                final List<Rating> chunk = ratingList.subList(i, Math.min(i + chunkSize, ratingList.size()));
                tasks.add(() -> {
                    for (Rating rating : chunk) {
                        String user = rating.getUserId();
                        double[] userVec = userFactors.get(user);
                        if (userVec == null) continue;

                        synchronized (userVec) {
                            for (String genre : rating.getGenres()) {
                                double[] genreVec = genreFactors.get(genre);
                                if (genreVec == null) continue;

                                double prediction = dot(userVec, genreVec);
                                double error = rating.getRating() - prediction;

                                synchronized (genreVec) {
                                    updateVectors(userVec, genreVec, error);
                                }
                            }
                        }
                    }
                });
            }

            // Usa CountDownLatch para esperar a conclusão da época atual
            CountDownLatch epochLatch = new CountDownLatch(tasks.size());
            for (Runnable task : tasks) {
                executor.execute(() -> {
                    try {
                        task.run();
                    } finally {
                        epochLatch.countDown();
                    }
                });
            }

            try {
                epochLatch.await();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

        executor.shutdown();
    }

    public static void updateVectors(double[] userVec, double[] genreVec, double error) {
        for (int i = 0; i < NUM_FEATURES; i++) {
            double u = userVec[i];
            double g = genreVec[i];
            userVec[i] += LEARNING_RATE * (error * g - REGULARIZATION * u);
            genreVec[i] += LEARNING_RATE * (error * u - REGULARIZATION * g);
        }
    }


    private static ConcurrentMap<String, Map<String, Double>> predictRatingsMatrix(ConcurrentLinkedQueue<Rating> ratings) {
        System.out.println("Iniciando geração da matriz com Executor e Runnable.");
        Map<String, List<Rating>> ratingsByUser = ratings.stream().collect(Collectors.groupingBy(Rating::getUserId));
        Map<String, List<String>> genresByTitle = ratings.stream()
                .collect(Collectors.toConcurrentMap(Rating::getTitle, Rating::getGenres, (e, r) -> e));

        ConcurrentMap<String, Map<String, Double>> matrix = new ConcurrentHashMap<>();
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        List<String> userList = new ArrayList<>(allUsers);
        CountDownLatch predictionLatch = new CountDownLatch(userList.size());

        for (String user : userList) {
            final String currentUser = user;
            executor.execute(() -> {
                try {
                    double[] userVec = userFactors.get(currentUser);
                    if (userVec != null) {
                        Map<String, Double> userRatings = new ConcurrentHashMap<>();
                        List<Rating> knownRatings = ratingsByUser.get(currentUser);
                        if (knownRatings != null) {
                            for (Rating r : knownRatings) {
                                userRatings.put(r.getTitle(), r.getRating());
                            }
                        }

                        genresByTitle.forEach((title, genres) -> {
                            if (!userRatings.containsKey(title)) {
                                double predicted = 0.0;
                                int validGenres = 0;
                                for (String genre : genres) {
                                    double[] genreVec = genreFactors.get(genre);
                                    if (genreVec != null) {
                                        predicted += dot(userVec, genreVec);
                                        validGenres++;
                                    }
                                }
                                if (validGenres > 0) {
                                    userRatings.put(title, predicted / validGenres);
                                }
                            }
                        });
                        matrix.put(currentUser, userRatings);
                    }
                } finally {
                    predictionLatch.countDown();
                }
            });
        }

        try {
            predictionLatch.await(); // Espera todas as predições terminarem
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        executor.shutdown();
        return matrix;
    }

    public static double dot(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();

        Set<String> arquivos = Set.of("dataset/avaliacoes_completas100MB.json");
        ConcurrentLinkedQueue<Rating> ratings = loadRatingsParallel(arquivos);

        long readTime = System.nanoTime();
        System.out.printf("lidos em: %.2f segundos%n", (readTime - startTime) / 1e9);

        initializeFactors(ratings);
        long initialTime = System.nanoTime();
        System.out.printf("initializeFactors rodou em: %.2f segundos%n", (initialTime - readTime) / 1e9);

        trainModel(ratings);
        long trainingTime = System.nanoTime();
        System.out.printf("trainingModel rodou em: %.2f segundos%n", (trainingTime - initialTime) / 1e9);

        ConcurrentMap<String, Map<String, Double>> matrix = predictRatingsMatrix(ratings);
        long matrixTime = System.nanoTime();
        System.out.printf("matrixGen rodou em: %.2f segundos%n", (matrixTime - trainingTime) / 1e9);

        long endTime = System.nanoTime();
        System.out.printf("Tempo total: %.2f segundos%n", (endTime - startTime) / 1e9);
    }
}
