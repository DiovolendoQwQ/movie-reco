const movieList = document.getElementById('movie-list');
const loadingIndicator = document.getElementById('loading-indicator');
const errorMessage = document.getElementById('error-message');
const pageTitle = document.getElementById('page-title');
// Use http://localhost:8000 if running API locally without docker-compose/nginx proxy
const API_BASE_URL = '/api'; // Assumes proxy in docker-compose, adjust if needed

function displayMovies(movies) {
  movieList.innerHTML = ''; // Clear previous movies or loading indicator
  loadingIndicator.style.display = 'none'; // Hide loading indicator
  errorMessage.classList.add('hidden'); // Hide error message

  if (!movies || movies.length === 0) {
    movieList.innerHTML = '<p class="text-gray-600 text-center col-span-full">未能找到相关推荐。</p>';
    return;
  }

  movies.forEach(movie => {
    const card = document.createElement('div');
    card.className = 'movie-card bg-white rounded-lg shadow-md overflow-hidden cursor-pointer';
    card.dataset.movieId = movie.movieId; // Store movie ID for click handling

    // Basic card structure - Title only for now
    // TODO: Add movie posters later (e.g., from OMDB API)
    card.innerHTML = `
            <div class="p-4">
                <h3 class="font-semibold text-lg text-gray-800 truncate" title="${movie.title}">${movie.title}</h3>
                <p class="text-sm text-gray-500">ID: ${movie.movieId}</p>
            </div>
        `;

    // Add click listener to each card
    card.addEventListener('click', () => {
      const movieId = card.dataset.movieId;
      // Redirect to the same page, but with the clicked movie ID as parameter
      window.location.href = `step.html?mid=${movieId}`;
    });

    movieList.appendChild(card);
  });
}

function showError(message = '无法加载推荐。请稍后重试或检查API服务。') {
  loadingIndicator.style.display = 'none';
  errorMessage.querySelector('span').textContent = message;
  errorMessage.classList.remove('hidden');
  movieList.innerHTML = ''; // Clear any potential partial content
}

async function fetchRecommendations() {
  const params = new URLSearchParams(window.location.search);
  const genre = params.get('g');
  const movieId = params.get('mid');

  let url = '';
  let options = {};

  if (genre) {
    pageTitle.textContent = `"${decodeURIComponent(genre)}" 类型电影推荐`;
    url = `${API_BASE_URL}/random?genre=${encodeURIComponent(genre)}`;
    options = { method: 'GET' };
  } else if (movieId) {
    pageTitle.textContent = '为你推荐更多相似电影';
    url = `${API_BASE_URL}/choice`;
    options = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Cookies should be sent automatically by the browser
      },
      body: JSON.stringify({ movieId: parseInt(movieId) }) // Send movie ID in body
    };
  } else {
    // Should not happen if navigated from index.html or a movie card
    showError('缺少必要的参数 (类型或电影ID)。');
    pageTitle.textContent = '推荐错误';
    return;
  }

  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      // Try to get error detail from API response body
      let errorDetail = `HTTP error! status: ${response.status}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorDetail = errorData.detail;
        }
      } catch (e) { /* Ignore JSON parsing error */ }
      throw new Error(errorDetail);
    }
    const data = await response.json();
    displayMovies(data.recommendations);
  } catch (error) {
    console.error('Error fetching recommendations:', error);
    showError(`加载推荐失败: ${error.message}`);
  }
}

// Fetch recommendations when the page loads
fetchRecommendations();
