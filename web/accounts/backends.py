"""Authentication backend that accepts either a username or an email address.

Django's default User model does not enforce unique emails, so an email that
belongs to more than one account is treated as ambiguous and rejected; those
users must log in with their username instead.
"""

from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend


class UsernameOrEmailBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        UserModel = get_user_model()
        if username is None:
            username = kwargs.get(UserModel.USERNAME_FIELD)
        if username is None or password is None:
            return None

        try:
            user = UserModel._default_manager.get_by_natural_key(username)
        except UserModel.DoesNotExist:
            matches = list(UserModel._default_manager.filter(email__iexact=username)[:2])
            user = matches[0] if len(matches) == 1 else None

        if user is None:
            # Run the hasher anyway so response time doesn't reveal whether the account exists.
            UserModel().set_password(password)
            return None
        if user.check_password(password) and self.user_can_authenticate(user):
            return user
        return None
