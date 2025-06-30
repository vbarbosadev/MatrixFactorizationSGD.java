package SGD;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import com.google.gson.*;
import com.google.gson.annotations.SerializedName;
import com.google.gson.reflect.TypeToken;
import com.google.gson.stream.JsonWriter;
import org.jetbrains.annotations.NotNull;


public class MovieRecommenderSGD {



    private static class Rating {
        String user_id;
        String title;

     
        @SerializedName(value = "genres", alternate = "genre")
        List<String> genre;

        double rating;

        public Rating(String user_id, String title, List<String> genre, double rating) {
            this.user_id = user_id;
            this.title = title;
            this.genre = genre;
            this.rating = rating;
        }

        public String getUserId() {
            return user_id;
        }

        public String getTitle() {
            return title;
        }

        public List<String> getGenres() {
            return genre;
        }

        public double getRating() {
            return rating;
        }
    }

    private static class RatingLoader {

        private static final Gson gson = new Gson();
        private static final Type ratingListType = new TypeToken<List<Rating>>() {}.getType();

        public static ConcurrentLinkedQueue<Rating> loadRatingsParallel(Set<String> filenames) {
            ConcurrentLinkedQueue<Rating> ratings = new ConcurrentLinkedQueue<>();
            List<Thread> threads = new ArrayList<>();

            for (String filename : filenames) {
                Thread t = new Thread(() -> {
                    try (FileReader reader = new FileReader(filename)) {
                        List<Rating> localList = gson.fromJson(reader, ratingListType);
                        ratings.addAll(localList);
                    } catch (IOException e) {
                        System.err.println("Erro ao ler arquivo: " + filename);
                        e.printStackTrace();
                    }
                });
                t.start(); 
                threads.add(t);
            }

            for (Thread t : threads) {
                try {
                    t.join();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }

            return ratings;
        }

        public static ConcurrentLinkedQueue<Rating> loadRatingsParallel(String filename) {
            return loadRatingsParallel(Set.of(filename));
        }
    }


    private static final int NUM_FEATURES = 10;
    private static final double LEARNING_RATE = 0.01;
    private static final double REGULARIZATION = 0.02;
    private static final int NUM_EPOCHS = 100;

    private static ConcurrentHashMap<String, double[]> userFactors = new ConcurrentHashMap<>();
    private static ConcurrentHashMap<String, double[]> genreFactors = new ConcurrentHashMap<>();
    private static Set<String> allGenres = ConcurrentHashMap.newKeySet();
    private static Set<String> allUsers = ConcurrentHashMap.newKeySet();


    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();

        Set<String> arquivos = Set.of("dataset/avaliacoes_completas1GB.json");

        ConcurrentLinkedQueue<Rating> ratings = RatingLoader.loadRatingsParallel(arquivos);

        long readTime = System.nanoTime();
        System.out.printf("lidos em: %.2f segundos%n", (readTime - startTime) / 1e9);

        initializeFactors(ratings);

        Map<String, List<String>> movieToGenresMap = new HashMap<>();
        for (Rating r : ratings) {
            allUsers.add(r.getUserId());
            allGenres.addAll(r.getGenres());
            movieToGenresMap.putIfAbsent(r.getTitle(), r.getGenres());
        }

        long initialTime = System.nanoTime();
        System.out.printf("initializeFactors rodou em: %.2f segundos%n", (initialTime - readTime) / 1e9);




        trainModel(ratings);

        long trainingTime = System.nanoTime();
        System.out.printf("trainingModel rodou em: %.2f segundos%n", (trainingTime - initialTime) / 1e9);


        ConcurrentMap<String, Map<String, Double>> matrix = predictRatingsMatrix(ratings);

        long matrixTime = System.nanoTime();
        System.out.printf("matrixGen rodou em: %.2f segundos%n", (matrixTime - trainingTime) / 1e9);



        long printTime = System.nanoTime();
        //  System.out.printf("printMatrix rodou em: %.2f segundos%n", (printTime - matrixTime) / 1e9);

        savePredictedRatingsToJson(matrix, allUsers, movieToGenresMap, "dataset/predicted_ratingsPlatform.json");

        long saveTime = System.nanoTime();
        System.out.printf("saveJson rodou em: %.2f segundos%n", (saveTime - printTime) / 1e9);


        long endTime = System.nanoTime();
        System.out.printf("Tempo total: %.2f segundos%n", (endTime - startTime) / 1e9);
    }


    private static void initializeFactors(@NotNull ConcurrentLinkedQueue<Rating> ratings) throws InterruptedException {
        allUsers = ConcurrentHashMap.newKeySet();
        allGenres = ConcurrentHashMap.newKeySet();

        ExecutorService executor1 = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        for (Rating r : ratings) {
            executor1.submit(() -> {
                allUsers.add(r.getUserId());
                allGenres.addAll(r.getGenres());
            });
        }

        executor1.shutdown();
        executor1.awaitTermination(1, TimeUnit.MINUTES);

        ExecutorService executor2 = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        for (String user : allUsers) {
            executor2.submit(() -> {
                double[] features = ThreadLocalRandom.current()
                        .doubles(NUM_FEATURES, 0, 0.1)
                        .toArray();
                userFactors.put(user, features);
            });
        }

        for (String genre : allGenres) {
            executor2.submit(() -> {
                double[] features = ThreadLocalRandom.current()
                        .doubles(NUM_FEATURES, 0, 0.1)
                        .toArray();
                genreFactors.put(genre, features);
            });
        }

        executor2.shutdown();
        executor2.awaitTermination(1, TimeUnit.MINUTES);
    }

    public static void trainModel(ConcurrentLinkedQueue<Rating> ratings) {
        System.out.println("Iniciando Treinamento");

        List<Rating> ratingList = new ArrayList<>(ratings);

        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);


        int chunkSize = (int) Math.ceil((double) ratingList.size() / (numThreads * 2));
        if (chunkSize == 0 && !ratingList.isEmpty()) {
            chunkSize = 1;
        }

        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            List<Callable<Void>> tasks = new ArrayList<>();

            for (int i = 0; i < ratingList.size(); i += chunkSize) {
                final int start = i;
                final int end = Math.min(start + chunkSize, ratingList.size());
                List<Rating> chunk = ratingList.subList(start, end);

                tasks.add(() -> {
                    for (Rating rating : chunk) {
                        String user = rating.getUserId();
                        double[] userVec = userFactors.get(user);

                        for (String genre : rating.getGenres()) {
                            double[] genreVec = genreFactors.get(genre);

                            double prediction = dot(userVec, genreVec);
                            double error = rating.getRating() - prediction;


                            updateVectors(userVec, genreVec, error);
                        }
                    }
                    return null;
                });
            }

            try {

                executor.invokeAll(tasks);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                System.err.println("O treinamento foi interrompido durante a época " + epoch);
                break;
            }
        }

        executor.shutdown();
    }


    private static void updateVectors(double[] userVec, double[] genreVec, double error) {
        for (int i = 0; i < NUM_FEATURES; i++) {
            double u = userVec[i];
            double g = genreVec[i];
            userVec[i] += LEARNING_RATE * (error * g - REGULARIZATION * u);
            genreVec[i] += LEARNING_RATE * (error * u - REGULARIZATION * g);
        }
    }


    private static ConcurrentMap<String, Map<String, Double>> predictRatingsMatrix(ConcurrentLinkedQueue<Rating> ratings)
            throws InterruptedException {

        System.out.println("Iniciando geração da matriz de predições...");

        ConcurrentMap<String, List<Rating>> ratingsByUser = new ConcurrentHashMap<>();
        ConcurrentMap<String, List<Rating>> ratingsByTitle = new ConcurrentHashMap<>();
        for (Rating r : ratings) {
            ratingsByUser.computeIfAbsent(r.getUserId(), k -> new ArrayList<>()).add(r);
            ratingsByTitle.computeIfAbsent(r.getTitle(), k -> new ArrayList<>()).add(r);
        }

        ConcurrentMap<String, Map<String, Double>> matrix = new ConcurrentHashMap<>();

        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        List<String> userList = new ArrayList<>(allUsers);
        if (userList.isEmpty() && !ratings.isEmpty()) {
            userList.addAll(ratingsByUser.keySet());
        }

        int chunkSize = (int) Math.ceil((double) userList.size() / (numThreads * 2));
        if (chunkSize == 0 && !userList.isEmpty()) {
            chunkSize = 1;
        }

        List<Callable<Void>> tasks = new ArrayList<>();

        for (int i = 0; i < userList.size(); i += chunkSize) {
            final int start = i;
            final int end = Math.min(start + chunkSize, userList.size());
            List<String> userChunk = userList.subList(start, end);

            tasks.add(() -> {
                for (String user : userChunk) {
                    double[] userVec = userFactors.get(user);
                    if (userVec == null) continue;

                    Map<String, Double> userRatings = new HashMap<>();

                    List<Rating> knownRatings = ratingsByUser.get(user);
                    if (knownRatings != null) {
                        for (Rating r : knownRatings) {
                            userRatings.put(r.getTitle(), r.getRating());
                        }
                    }

                    for (Map.Entry<String, List<Rating>> entry : ratingsByTitle.entrySet()) {
                        String title = entry.getKey();

                        if (userRatings.containsKey(title)) continue;

                        List<Rating> ratingsForTitle = entry.getValue();
                        if (ratingsForTitle.isEmpty()) continue;

                        List<String> currentGenres = ratingsForTitle.get(0).getGenres();
                        if (currentGenres == null || currentGenres.isEmpty()) continue;

                        double predictedScore = 0.0;
                        int validGenres = 0;
                        for (String genre : currentGenres) {
                            double[] genreVec = genreFactors.get(genre);
                            if (genreVec != null) {
                                predictedScore += dot(userVec, genreVec);
                                validGenres++;
                            }
                        }

                        if (validGenres > 0) {
                            userRatings.put(title, predictedScore / validGenres);
                        }
                    }
                    matrix.put(user, userRatings);
                }
                return null;
            });
        }

        if (!tasks.isEmpty()) {
            executor.invokeAll(tasks);
        }
        executor.shutdown();

        System.out.println("Matriz de predições gerada com sucesso.");
        return matrix;
    }

    
    public static void savePredictedRatingsToJson(Map<String, Map<String, Double>> predictedMatrix,
                                                  Set<String> users,
                                                  Map<String, List<String>> genreMap,
                                                  String filename) {
        List<String> allUsersOrdered = new ArrayList<>(users);
        Collections.sort(allUsersOrdered);

        List<String> allTitlesOrdered = new ArrayList<>(genreMap.keySet());
        Collections.sort(allTitlesOrdered);

        try (JsonWriter writer = new JsonWriter(new FileWriter(filename))) {
            writer.setIndent("  ");

            writer.beginArray();

            for (String user : allUsersOrdered) {
                writer.beginObject();
                writer.name("user_id").value(user);

                writer.name("movies");
                writer.beginArray();

                Map<String, Double> userRatings = predictedMatrix.getOrDefault(user, Collections.emptyMap());

                for (String title : allTitlesOrdered) {
                    writer.beginObject();
                    writer.name("title").value(title);

                    writer.name("genre");
                    writer.beginArray();
                    List<String> genres = genreMap.getOrDefault(title, Collections.emptyList());
                    for (String genre : genres) {
                        writer.value(genre);
                    }
                    writer.endArray();

                    writer.name("rating");
                    Double rating = userRatings.get(title);
                    if (rating != null) {
                        writer.value(rating);
                    } else {
                        writer.nullValue();
                    }
                    writer.endObject();
                }
                writer.endArray();
                writer.endObject();
            }
            writer.endArray();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    private static double dot(double[] a, double[] b) {
        final int length = a.length;
        double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        int i = 0;
        for (; i <= length - 4; i += 4) {
            sum0 += a[i] * b[i];
            sum1 += a[i + 1] * b[i + 1];
            sum2 += a[i + 2] * b[i + 2];
            sum3 += a[i + 3] * b[i + 3];
        }
        double sum = sum0 + sum1 + sum2 + sum3;
        for (; i < length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

}
